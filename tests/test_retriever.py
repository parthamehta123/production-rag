"""Tests for hybrid retriever logic."""

from unittest.mock import MagicMock, patch

from src.retriever import HybridRetriever, detect_companies_in_query


def test_detect_companies_single():
    assert detect_companies_in_query("What are Apple's risk factors?") == ["AAPL"]
    assert detect_companies_in_query("Tell me about Tesla") == ["TSLA"]
    assert detect_companies_in_query("JPMorgan business segments") == ["JPM"]


def test_detect_companies_multiple():
    result = detect_companies_in_query("Compare Apple and Microsoft")
    assert "AAPL" in result
    assert "MSFT" in result


def test_detect_companies_none():
    assert detect_companies_in_query("What is the weather today?") == []
    assert detect_companies_in_query("Tell me about regulatory compliance") == []


def _make_retriever():
    """Create a HybridRetriever with mocked external dependencies."""
    with patch.object(HybridRetriever, "__init__", lambda self, **kw: None):
        retriever = HybridRetriever()
        retriever.top_k = 10
        retriever.rerank_top_k = 5
        retriever.documents = [
            "[Apple Inc. (AAPL) - Risk Factors] The return policy allows returns within 30 days.",
            "[Apple Inc. (AAPL) - Business] Shipping takes 5 to 7 business days.",
            "[Tesla, Inc. (TSLA) - Business] Contact support at help@example.com.",
        ]
        retriever.metadatas = [
            {"source": "apple_10k.txt", "company": "Apple Inc.", "ticker": "AAPL"},
            {"source": "apple_10k.txt", "company": "Apple Inc.", "ticker": "AAPL"},
            {"source": "tesla_10k.txt", "company": "Tesla, Inc.", "ticker": "TSLA"},
        ]
        retriever.ids = ["1", "2", "3"]
        retriever.company_indices = {"AAPL": [0, 1], "TSLA": [2]}

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


def test_bm25_search_with_company_filter():
    retriever = _make_retriever()
    # Only search AAPL chunks
    results = retriever._bm25_search("return policy", k=3, allowed_indices=[0, 1])
    for r in results:
        assert "AAPL" in r["content"] or "Apple" in r["content"]


def test_deduplicate_removes_duplicates():
    retriever = _make_retriever()
    results = [
        {"content": "Same content here", "score": 0.8, "source": "vector"},
        {"content": "Same content here", "score": 0.9, "source": "bm25"},
        {"content": "Different content", "score": 0.7, "source": "vector"},
    ]
    deduped = retriever._deduplicate(results)
    assert len(deduped) == 2
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
