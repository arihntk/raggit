"""LLM-as-judge utilities for evaluating answer quality."""

from __future__ import annotations

import json
import re

from raggit.llm.base import LLMProvider

JUDGE_PROMPT = """You are an expert evaluator for retrieval-augmented generation systems.

Evaluate the following answer to the user question.
Score the answer on a scale of 0 to 5 where:
- 5: completely correct, well-supported, and directly answers the question
- 4: mostly correct with minor issues
- 3: partially correct or incomplete
- 2: largely incorrect but contains some relevant information
- 1: mostly incorrect or hallucinated
- 0: completely incorrect, irrelevant, or harmful

Respond in JSON format with exactly two keys:
- "score": integer 0-5
- "reasoning": a brief explanation of the score

Question: {question}

Expected answer (ground truth): {expected_answer}

Actual answer: {actual_answer}

JSON response:"""


_GROUNDEDNESS_PROMPT = """You are evaluating whether an answer is grounded in the provided context.

Context:
{context}

Question: {question}
Answer: {answer}

Is the answer fully supported by the context? Respond with only "YES" or "NO".
"""


def _extract_json(text: str) -> dict[str, object]:
    """Best-effort extraction of a JSON object from a string."""
    # Try to find a JSON block between braces.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


async def judge_answer(
    llm: LLMProvider,
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> tuple[float, str]:
    """Ask an LLM to score an answer against ground truth.

    Returns a normalized score in [0, 1] and reasoning text.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer or "",
    )
    response = await llm.generate(
        system_prompt=None,
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=512,
    )
    parsed = _extract_json(response)
    score = parsed.get("score")
    reasoning = str(parsed.get("reasoning", ""))
    if isinstance(score, (int, float)):
        normalized = max(0.0, min(1.0, float(score) / 5.0))
    else:
        normalized = 0.0
        reasoning = reasoning or "Could not parse judge response."
    return normalized, reasoning


async def judge_groundedness(
    llm: LLMProvider,
    question: str,
    answer: str,
    context: str,
) -> bool:
    """Ask an LLM whether the answer is grounded in the provided context."""
    prompt = _GROUNDEDNESS_PROMPT.format(
        context=context,
        question=question,
        answer=answer or "",
    )
    response = await llm.generate(
        system_prompt=None,
        user_prompt=prompt,
        temperature=0.0,
        max_tokens=16,
    )
    return response.strip().upper().startswith("YES")
