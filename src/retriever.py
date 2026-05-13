"""Hybrid retriever: combines BM25 keyword search with vector semantic search."""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()


class HybridRetriever:
    """Combines BM25 keyword search + vector semantic search with cross-encoder reranking."""

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        top_k: int = 10,
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

        # BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Cross-encoder reranker
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _vector_search(self, query: str, k: int) -> list[dict]:
        """Semantic search using vector embeddings."""
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
                "source": "vector",
            }
            for doc, score in results
        ]

    def _bm25_search(self, query: str, k: int) -> list[dict]:
        """Keyword search using BM25."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
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
        """Run hybrid retrieval: BM25 + vector search, deduplicate, then rerank."""
        vector_results = self._vector_search(query, self.top_k)
        bm25_results = self._bm25_search(query, self.top_k)

        combined = vector_results + bm25_results
        deduped = self._deduplicate(combined)
        reranked = self._rerank(query, deduped)

        # Add source IDs for citation
        for i, r in enumerate(reranked):
            r["source_id"] = i + 1

        return reranked
