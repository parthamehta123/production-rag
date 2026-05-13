"""Tests for hybrid retriever logic."""

from unittest.mock import MagicMock, patch

from src.retriever import HybridRetriever


def _make_retriever():
    """Create a HybridRetriever with mocked external dependencies."""
    with patch.object(HybridRetriever, "__init__", lambda self, **kw: None):
        retriever = HybridRetriever()
        retriever.top_k = 5
        retriever.rerank_top_k = 3
        retriever.documents = [
            "The return policy allows returns within 30 days.",
            "Shipping takes 5 to 7 business days.",
            "Contact support at help@example.com.",
        ]
        retriever.metadatas = [{"source": "faq.txt"}] * 3
        retriever.ids = ["1", "2", "3"]

        from rank_bm25 import BM25Okapi

        tokenized = [doc.lower().split() for doc in retriever.documents]
        retriever.bm25 = BM25Okapi(tokenized)

        retriever.reranker = MagicMock()
        retriever.reranker.predict = MagicMock(return_value=[0.9, 0.5, 0.1])
        retriever.vectorstore = MagicMock()
        return retriever


def test_bm25_search_returns_results():
    retriever = _make_retriever()
    results = retriever._bm25_search("return policy", k=3)
    assert len(results) > 0
    assert results[0]["source"] == "bm25"
    assert "return" in results[0]["content"].lower()


def test_deduplicate_removes_duplicates():
    retriever = _make_retriever()
    results = [
        {"content": "Same content here", "score": 0.8, "source": "vector"},
        {"content": "Same content here", "score": 0.9, "source": "bm25"},
        {"content": "Different content", "score": 0.7, "source": "vector"},
    ]
    deduped = retriever._deduplicate(results)
    assert len(deduped) == 2
    # Should keep the higher score
    same_content = [r for r in deduped if r["content"] == "Same content here"][0]
    assert same_content["score"] == 0.9


def test_rerank_limits_to_top_k():
    retriever = _make_retriever()
    retriever.rerank_top_k = 2
    results = [
        {"content": "doc1", "score": 0.5},
        {"content": "doc2", "score": 0.6},
        {"content": "doc3", "score": 0.7},
    ]
    reranked = retriever._rerank("test query", results)
    assert len(reranked) == 2


def test_rerank_empty_input():
    retriever = _make_retriever()
    assert retriever._rerank("test", []) == []
