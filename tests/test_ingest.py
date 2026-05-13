"""Tests for document ingestion pipeline."""

from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.ingest import chunk_documents


def test_chunk_documents_splits_correctly():
    docs = [Document(page_content="word " * 200, metadata={"source": "test.txt"})]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 120  # allow slight overflow from splitter


def test_chunk_documents_preserves_metadata():
    docs = [
        Document(
            page_content="Hello world. " * 50, metadata={"source": "a.pdf", "page": 1}
        )
    ]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["source"] == "a.pdf"


def test_chunk_documents_empty_input():
    chunks = chunk_documents([], chunk_size=100, chunk_overlap=20)
    assert chunks == []
