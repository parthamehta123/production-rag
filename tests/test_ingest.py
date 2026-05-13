"""Tests for document ingestion pipeline."""

from langchain_core.documents import Document

from src.ingest import chunk_documents, detect_company, detect_section


def test_chunk_documents_splits_correctly():
    docs = [
        Document(page_content="word " * 200, metadata={"source": "apple_10k_2025.txt"})
    ]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 1


def test_chunk_documents_enriches_metadata():
    docs = [
        Document(
            page_content="Item 1A. Risk Factors. The company faces significant risks. "
            * 10,
            metadata={"source": "data/documents/apple_10k_2025.txt"},
        )
    ]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["company"] == "Apple Inc."
        assert chunk.metadata["ticker"] == "AAPL"
        assert chunk.metadata["section"] == "Risk Factors"
        # Chunk should have company prefix
        assert "[Apple Inc. (AAPL)" in chunk.page_content


def test_chunk_documents_filters_short_fragments():
    docs = [
        Document(
            page_content="Short header", metadata={"source": "apple_10k_2025.txt"}
        ),
        Document(
            page_content="This is a long enough paragraph that should survive the filter. "
            * 5,
            metadata={"source": "apple_10k_2025.txt"},
        ),
    ]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=20)
    # Short fragment should be filtered out
    for chunk in chunks:
        assert len(chunk.page_content) > 80


def test_chunk_documents_empty_input():
    chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
    assert chunks == []


def test_detect_company():
    assert detect_company("data/documents/apple_10k_2025.txt")["ticker"] == "AAPL"
    assert detect_company("data/documents/tesla_10k_2025.txt")["ticker"] == "TSLA"
    assert detect_company("data/documents/jpmorgan_10k_2025.txt")["ticker"] == "JPM"
    assert detect_company("unknown_file.txt")["ticker"] == "N/A"


def test_detect_section():
    assert detect_section("Item 1A. Risk Factors") == "Risk Factors"
    assert detect_section("Item 1. Business overview") == "Business"
    assert detect_section("Competition in the market") == "Competition"
    assert detect_section("Nothing special here") == "General"
