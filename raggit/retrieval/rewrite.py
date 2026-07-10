"""Query rewriting: multi-query expansion and HyDE."""

from __future__ import annotations

from raggit.api.models import QueryRewriteMode
from raggit.core.logging import get_logger
from raggit.llm.base import LLMProvider

logger = get_logger("raggit.retrieval.rewrite")


async def expand_multi_query(
    llm: LLMProvider,
    query: str,
    count: int = 3,
) -> list[str]:
    """Generate alternative phrasings of the query for broader recall."""
    prompt = (
        f"Generate {count} diverse alternative search queries for the following "
        "question. Return only the queries, one per line, no numbering.\n\n"
        f"Question: {query}"
    )
    try:
        text = await llm.generate(
            system_prompt=(
                "You rewrite user questions into search queries. "
                "Output only queries, one per line."
            ),
            user_prompt=prompt,
            temperature=0.4,
            max_tokens=256,
        )
    except Exception:
        logger.exception("Multi-query expansion failed")
        return [query]

    lines = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    # Keep unique, preserve order, always include original first
    seen: set[str] = {query.lower()}
    results = [query]
    for line in lines:
        key = line.lower()
        if key not in seen:
            seen.add(key)
            results.append(line)
        if len(results) >= count + 1:
            break
    return results


async def hyde_document(
    llm: LLMProvider,
    query: str,
) -> str:
    """Hypothetical Document Embeddings: generate a fake answer passage to embed."""
    prompt = (
        "Write a short factual paragraph that would answer the following question. "
        "Do not mention that this is hypothetical. Be concrete.\n\n"
        f"Question: {query}"
    )
    try:
        return await llm.generate(
            system_prompt="You write concise encyclopedic passages.",
            user_prompt=prompt,
            temperature=0.3,
            max_tokens=400,
        )
    except Exception:
        logger.exception("HyDE generation failed")
        return query


async def rewrite_queries(
    mode: QueryRewriteMode,
    query: str,
    llm: LLMProvider | None,
    multi_query_count: int = 3,
) -> tuple[list[str], str | None]:
    """Return (queries_for_keyword_search, optional_hyde_passage_for_embedding).

    For NONE: ([query], None)
    For MULTI_QUERY: (expanded queries, None)
    For HYDE: ([query], hypothetical passage)
    """
    if mode == QueryRewriteMode.NONE or llm is None:
        return [query], None

    if mode == QueryRewriteMode.MULTI_QUERY:
        queries = await expand_multi_query(llm, query, count=multi_query_count)
        return queries, None

    if mode == QueryRewriteMode.HYDE:
        passage = await hyde_document(llm, query)
        return [query], passage

    return [query], None
