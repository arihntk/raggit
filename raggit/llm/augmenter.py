"""Prompt augmentation for RAG."""

from __future__ import annotations

from raggit.api.models import QueryResult

SYSTEM_PROMPT = """You are a precise retrieval-augmented assistant.
Answer the user's question using only the provided context.
If the context does not contain enough information, say so clearly.
Do not make up facts. Cite the source chunks implicitly by referencing their content."""


def build_augmented_prompt(result: QueryResult) -> str:
    """Build a prompt from the original query, keywords, and retrieved chunks."""
    lines: list[str] = []
    lines.append("## Original Question")
    lines.append(result.query)
    lines.append("")

    if result.sanitized_keywords:
        lines.append("## Extracted Keywords")
        lines.append(", ".join(result.sanitized_keywords))
        lines.append("")

    lines.append("## Context")
    for i, retrieved in enumerate(result.chunks, start=1):
        lines.append(f"### Chunk {i}")
        lines.append(retrieved.chunk.cleaned_content)
        lines.append("")

    lines.append("## Instructions")
    lines.append(
        "Answer the original question based on the context above. "
        "Be concise and accurate."
    )

    return "\n".join(lines)


async def augment_and_answer(
    llm: object,
    result: QueryResult,
    system_prompt: str | None = None,
) -> str:
    """Augment retrieved chunks with the query and ask the LLM."""
    from raggit.llm.base import LLMProvider

    if not isinstance(llm, LLMProvider):
        msg = f"Expected LLMProvider, got {type(llm).__name__}"
        raise TypeError(msg)

    user_prompt = build_augmented_prompt(result)
    answer = await llm.generate(
        system_prompt=system_prompt or SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    return str(answer)
