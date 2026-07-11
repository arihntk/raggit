"""Tests for query rewriting (multi-query and HyDE)."""

from raggit.api.models import QueryRewriteMode
from raggit.retrieval.rewrite import (
    expand_multi_query,
    hyde_document,
    rewrite_queries,
)


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response

    async def generate(
        self,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.response


async def test_expand_multi_query_returns_original_plus_variants() -> None:
    llm = FakeLLM(
        "alternative one\nalternative two\nalternative three\n"
    )
    queries = await expand_multi_query(llm, "what is raggit", count=3)
    assert queries[0] == "what is raggit"
    assert len(queries) == 4
    assert "alternative one" in queries


async def test_hyde_document_returns_generated_passage() -> None:
    llm = FakeLLM("raggit is a retrieval-augmented generation system.")
    passage = await hyde_document(llm, "what is raggit")
    assert "retrieval-augmented" in passage


async def test_rewrite_queries_none() -> None:
    queries, hyde = await rewrite_queries(QueryRewriteMode.NONE, "what is raggit", None)
    assert queries == ["what is raggit"]
    assert hyde is None


async def test_rewrite_queries_multi_query() -> None:
    llm = FakeLLM("variant one\nvariant two")
    queries, hyde = await rewrite_queries(
        QueryRewriteMode.MULTI_QUERY, "what is raggit", llm, multi_query_count=2
    )
    assert hyde is None
    assert queries[0] == "what is raggit"
    assert "variant one" in queries


async def test_rewrite_queries_hyde() -> None:
    llm = FakeLLM("raggit is a RAG system.")
    queries, hyde = await rewrite_queries(QueryRewriteMode.HYDE, "what is raggit", llm)
    assert queries == ["what is raggit"]
    assert hyde is not None
    assert "RAG system" in hyde


async def test_rewrite_queries_hyde_without_llm_returns_original() -> None:
    queries, hyde = await rewrite_queries(QueryRewriteMode.HYDE, "what is raggit", None)
    assert queries == ["what is raggit"]
    assert hyde is None
