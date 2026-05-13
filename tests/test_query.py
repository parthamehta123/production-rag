"""Tests for query pipeline."""

from unittest.mock import MagicMock, patch

from src.query import format_context, load_prompts


def test_format_context():
    chunks = [
        {"source_id": 1, "content": "First chunk content"},
        {"source_id": 2, "content": "Second chunk content"},
    ]
    result = format_context(chunks)
    assert "[Source 1]" in result
    assert "[Source 2]" in result
    assert "First chunk content" in result
    assert "Second chunk content" in result


def test_load_prompts():
    prompts = load_prompts("configs/prompts.yaml")
    assert "system_prompt" in prompts
    assert "query_template" in prompts
    assert "{context}" in prompts["query_template"]
    assert "{question}" in prompts["query_template"]


def test_load_prompts_has_citation_format():
    prompts = load_prompts("configs/prompts.yaml")
    assert "citation_format" in prompts
    assert "source_id" in prompts["citation_format"]
