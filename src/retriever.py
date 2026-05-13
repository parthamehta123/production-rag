"""Hybrid retriever: combines BM25 keyword search with vector semantic search.

Key architectural decisions:
- Company detection from query to filter metadata before retrieval
- Wider initial retrieval (top_k=20) narrowed by reranker to top 5
- BM25 filtered by company when a specific company is detected
- Reranker sees more candidates, producing better final results
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

# Company name variants for query detection
COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft", "azure"],
    "TSLA": ["tesla", "tsla"],
    "JPM": ["jpmorgan", "jpm", "jp morgan", "chase"],
    "GS": ["goldman", "goldman sachs", "gs"],
}


def detect_companies_in_query(query: str) -> list[str]:
    """Detect which companies are mentioned in the query. Returns list of tickers."""
    query_lower = query.lower()
    found = []
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(alias in query_lower for alias in aliases):
            found.append(ticker)
    return found


class HybridRetriever:
    """Combines BM25 keyword search + vector semantic search with cross-encoder reranking."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        top_k: int = 20,
        rerank_top_k: int = 5,
    ):
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k

        # Vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )

        # Load all documents for BM25
        collection = self.vectorstore.get()
        self.documents = collection["documents"]
        self.metadatas = collection["metadatas"]
        self.ids = collection["ids"]

        # Build index mapping company tickers to document indices for filtered BM25
        self.company_indices: dict[str, list[int]] = {}
        for i, meta in enumerate(self.metadatas):
            ticker = meta.get("ticker", "")
            if ticker not in self.company_indices:
                self.company_indices[ticker] = []
            self.company_indices[ticker].append(i)

        # BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _vector_search(
        self, query: str, k: int, company_filter: dict | None = None
    ) -> list[dict]:
        """Semantic search using vector embeddings with optional metadata filter."""
        search_kwargs = {"k": k}
        if company_filter:
            search_kwargs["filter"] = company_filter

        results = self.vectorstore.similarity_search_with_score(query, **search_kwargs)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "source": "vector",
            }
            for doc, score in results
        ]

    def _bm25_search(
        self, query: str, k: int, allowed_indices: list[int] | None = None
    ) -> list[dict]:
        """Keyword search using BM25, optionally restricted to specific document indices."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        if allowed_indices is not None:
            # Zero out scores for documents not in allowed set
            mask = set(allowed_indices)
            for i in range(len(scores)):
                if i not in mask:
                    scores[i] = 0

        top_indices = scores.argsort()[-k:][::-1]
        return [
            {
                "content": self.documents[i],
                "metadata": self.metadatas[i] if self.metadatas else {},
                "score": float(scores[i]),
                "source": "bm25",
            }
            for i in top_indices
            if scores[i] > 0
        ]

    def _deduplicate(self, results: list[dict]) -> list[dict]:
        """Remove duplicate chunks, keeping the highest-scored version."""
        seen = {}
        for r in results:
            key = r["content"][:200]
            if key not in seen or r["score"] > seen[key]["score"]:
                seen[key] = r
        return list(seen.values())

    def _rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Rerank results using a cross-encoder model."""
        if not results:
            return []
        pairs = [(query, r["content"]) for r in results]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[: self.rerank_top_k]

    def retrieve(self, query: str) -> list[dict]:
        """Run hybrid retrieval with company-aware filtering.

        Pipeline:
        1. Detect companies mentioned in query
        2. If single company → filter both vector and BM25 to that company's chunks
        3. If cross-company or no company → search all chunks
        4. Deduplicate across both search methods
        5. Rerank with cross-encoder
        """
        companies = detect_companies_in_query(query)

        company_filter = None
        bm25_allowed = None

        if len(companies) == 1:
            # Single company query → metadata-filtered retrieval
            ticker = companies[0]
            company_filter = {"ticker": ticker}
            bm25_allowed = self.company_indices.get(ticker, None)

        vector_results = self._vector_search(query, self.top_k, company_filter)
        bm25_results = self._bm25_search(query, self.top_k, bm25_allowed)

        combined = vector_results + bm25_results
        deduped = self._deduplicate(combined)
        reranked = self._rerank(query, deduped)

        # Add source IDs for citation
        for i, r in enumerate(reranked):
            r["source_id"] = i + 1

        return reranked
