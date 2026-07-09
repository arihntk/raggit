"""Tests for query sanitization."""

from raggit.retrieval.sanitizer import build_bm25_query, sanitize_query


def test_sanitize_query_extracts_keywords() -> None:
    query = "What is the best way to use raggit for RAG?"
    cleaned, keywords = sanitize_query(query)
    assert cleaned == query
    assert "raggit" in keywords
    assert "best" in keywords
    assert "way" in keywords
    assert "the" not in keywords
    assert "is" not in keywords


def test_sanitize_query_empty() -> None:
    cleaned, keywords = sanitize_query("is the a an")
    assert cleaned == "is the a an"
    assert keywords == []


def test_build_bm25_query() -> None:
    keywords = ["raggit", "production", "rag"]
    assert build_bm25_query(keywords) == "raggit & production & rag"
