"""raggit query command."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from raggit.api.models import Citation, QueryFilters, QueryResult, QueryRewriteMode
from raggit.cli.options import query_rewrite_option
from raggit.core.config import get_settings
from raggit.core.logging import configure_logging
from raggit.db.session import AsyncSessionLocal

console = Console()


def _apply_overrides(
    config: Any,
    *,
    top_k: int | None,
    min_top_k: int | None,
    max_top_k: int | None,
    top_k_ratio: float | None,
    rrf_k: int | None,
    parent_window: int | None,
    min_score: float | None,
    rewrite: QueryRewriteMode | None,
    multi_query_count: int | None,
    reranker_enabled: bool | None,
    reranker_model: str | None,
    reranker_top_n: int | None,
    refuse_on_empty: bool | None,
    refuse_on_low_score: bool | None,
    min_answer_score: float | None,
    groundedness_check: bool | None,
    pii_redaction: bool | None,
    prompt_injection_hardening: bool | None,
    log_level: str | None,
) -> None:
    """Apply CLI overrides onto the loaded RAGConfig."""
    if log_level is not None:
        config.log_level = log_level
    if top_k is not None:
        config.retrieval.min_top_k = top_k
        config.retrieval.max_top_k = top_k
        config.retrieval.top_k_ratio = 0.0
        config.min_top_k = top_k
        config.max_top_k = top_k
        config.top_k_ratio = 0.0
    else:
        if min_top_k is not None:
            config.retrieval.min_top_k = min_top_k
            config.min_top_k = min_top_k
        if max_top_k is not None:
            config.retrieval.max_top_k = max_top_k
            config.max_top_k = max_top_k
        if top_k_ratio is not None:
            config.retrieval.top_k_ratio = top_k_ratio
            config.top_k_ratio = top_k_ratio
    if rrf_k is not None:
        config.retrieval.rrf_k = rrf_k
        config.rrf_k = rrf_k
    if parent_window is not None:
        config.retrieval.parent_window = parent_window
    if min_score is not None:
        config.retrieval.min_score = min_score
    if rewrite is not None:
        config.retrieval.query_rewrite = rewrite
    if multi_query_count is not None:
        config.retrieval.multi_query_count = multi_query_count
    if reranker_enabled is not None:
        config.retrieval.reranker.enabled = reranker_enabled
    if reranker_model is not None:
        config.retrieval.reranker.model = reranker_model
    if reranker_top_n is not None:
        config.retrieval.reranker.top_n = reranker_top_n
    if refuse_on_empty is not None:
        config.safety.refuse_on_empty = refuse_on_empty
    if refuse_on_low_score is not None:
        config.safety.refuse_on_low_score = refuse_on_low_score
    if min_answer_score is not None:
        config.safety.min_answer_score = min_answer_score
    if groundedness_check is not None:
        config.safety.groundedness_check = groundedness_check
    if pii_redaction is not None:
        config.safety.pii_redaction = pii_redaction
    if prompt_injection_hardening is not None:
        config.safety.prompt_injection_hardening = prompt_injection_hardening


def _format_retrieved_table(result: QueryResult) -> Table:
    """Render a Rich table for retrieved chunks."""
    table = Table(title="Retrieved Chunks")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source")
    table.add_column("Chunk")
    for rank, retrieved in enumerate(result.chunks, start=1):
        source = retrieved.chunk.filename or retrieved.chunk.source_uri or ""
        table.add_row(
            str(rank),
            f"{retrieved.score:.4f}",
            source[:40],
            retrieved.chunk.cleaned_content[:200],
        )
    return table


def _format_citation_tree(citations: list[Citation]) -> Tree:
    """Render a Rich tree for citations."""
    tree = Tree("[bold]Sources[/bold]")
    for i, cite in enumerate(citations, start=1):
        loc = cite.filename or cite.source_uri or str(cite.chunk_id)
        extras: list[str] = []
        if cite.page_number is not None:
            extras.append(f"page {cite.page_number}")
        if cite.section_title:
            extras.append(cite.section_title)
        if cite.start_offset is not None and cite.end_offset is not None:
            extras.append(f"offsets {cite.start_offset}-{cite.end_offset}")
        suffix = f" ({', '.join(extras)})" if extras else ""
        tree.add(f"[{i}] {loc}{suffix} — chunk {cite.chunk_index}")
    return tree


def register_query(app: typer.Typer) -> None:
    """Register the query command with the CLI application."""

    @app.command()
    def query(
        question: str = typer.Argument(..., help="Question to ask"),
        top_k: int | None = typer.Option(
            None, "--top-k", help="Fixed number of chunks to retrieve"
        ),
        min_top_k: int | None = typer.Option(
            None, "--min-top-k", help="Minimum number of chunks to retrieve"
        ),
        max_top_k: int | None = typer.Option(
            None, "--max-top-k", help="Maximum number of chunks to retrieve"
        ),
        top_k_ratio: float | None = typer.Option(
            None, "--top-k-ratio", help="Fraction of total chunks used to scale top-k"
        ),
        rrf_k: int | None = typer.Option(None, "--rrf-k", help="Reciprocal rank fusion constant"),
        source_prefix: str | None = typer.Option(
            None, "--source-prefix", help="Filter by source URI prefix"
        ),
        filename_prefix: str | None = typer.Option(
            None, "--filename-prefix", help="Filter by filename prefix"
        ),
        tenant: str | None = typer.Option(None, "--tenant", help="Filter by tenant id"),
        tag: list[str] = typer.Option(None, "--tag", help="Filter by tag (repeatable)"),
        document_id: list[str] = typer.Option(
            None, "--document-id", help="Filter by document UUID (repeatable)"
        ),
        created_after: datetime | None = typer.Option(
            None, "--created-after", help="Filter documents created after this ISO timestamp"
        ),
        created_before: datetime | None = typer.Option(
            None, "--created-before", help="Filter documents created before this ISO timestamp"
        ),
        min_score: float | None = typer.Option(
            None, "--min-score", help="Drop chunks below this score"
        ),
        rewrite: QueryRewriteMode = query_rewrite_option(),
        multi_query_count: int | None = typer.Option(
            None, "--multi-query-count", help="Number of variants for multi_query rewrite"
        ),
        parent_window: int | None = typer.Option(
            None, "--parent-window", help="Expand hits by +/- N sibling chunks"
        ),
        reranker_enabled: bool | None = typer.Option(
            None, "--reranker/--no-reranker", help="Enable cross-encoder reranking"
        ),
        reranker_model: str | None = typer.Option(
            None, "--reranker-model", help="Cross-encoder model name"
        ),
        reranker_top_n: int | None = typer.Option(
            None, "--reranker-top-n", help="Number of candidates to rerank"
        ),
        refuse_on_empty: bool | None = typer.Option(
            None,
            "--refuse-on-empty/--no-refuse-on-empty",
            help="Refuse when no chunks are retrieved",
        ),
        refuse_on_low_score: bool | None = typer.Option(
            None,
            "--refuse-on-low-score/--no-refuse-on-low-score",
            help="Refuse when scores are below threshold",
        ),
        min_answer_score: float | None = typer.Option(
            None, "--min-answer-score", help="Minimum score required for an answer"
        ),
        groundedness_check: bool | None = typer.Option(
            None,
            "--groundedness-check/--no-groundedness-check",
            help="Enable groundedness check",
        ),
        pii_redaction: bool | None = typer.Option(
            None, "--pii-redaction/--no-pii-redaction", help="Redact PII before embedding"
        ),
        prompt_injection_hardening: bool | None = typer.Option(
            None,
            "--prompt-injection-hardening/--no-prompt-injection-hardening",
            help="Harden chunks against prompt injection",
        ),
        no_llm: bool = typer.Option(
            False, "--no-llm", help="Show retrieved chunks only; do not generate an answer"
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
    ) -> None:
        """Ask a question against the indexed documents."""
        asyncio.run(
            _query(
                question,
                top_k=top_k,
                min_top_k=min_top_k,
                max_top_k=max_top_k,
                top_k_ratio=top_k_ratio,
                rrf_k=rrf_k,
                source_prefix=source_prefix,
                filename_prefix=filename_prefix,
                tenant=tenant,
                tags=tag,
                document_ids=document_id,
                created_after=created_after,
                created_before=created_before,
                min_score=min_score,
                rewrite=rewrite,
                multi_query_count=multi_query_count,
                parent_window=parent_window,
                reranker_enabled=reranker_enabled,
                reranker_model=reranker_model,
                reranker_top_n=reranker_top_n,
                refuse_on_empty=refuse_on_empty,
                refuse_on_low_score=refuse_on_low_score,
                min_answer_score=min_answer_score,
                groundedness_check=groundedness_check,
                pii_redaction=pii_redaction,
                prompt_injection_hardening=prompt_injection_hardening,
                no_llm=no_llm,
                log_level=log_level,
            )
        )


async def _query(
    question: str,
    *,
    top_k: int | None,
    min_top_k: int | None,
    max_top_k: int | None,
    top_k_ratio: float | None,
    rrf_k: int | None,
    source_prefix: str | None,
    filename_prefix: str | None,
    tenant: str | None,
    tags: list[str] | None,
    document_ids: list[str] | None,
    created_after: datetime | None,
    created_before: datetime | None,
    min_score: float | None,
    rewrite: QueryRewriteMode,
    multi_query_count: int | None,
    parent_window: int | None,
    reranker_enabled: bool | None,
    reranker_model: str | None,
    reranker_top_n: int | None,
    refuse_on_empty: bool | None,
    refuse_on_low_score: bool | None,
    min_answer_score: float | None,
    groundedness_check: bool | None,
    pii_redaction: bool | None,
    prompt_injection_hardening: bool | None,
    no_llm: bool,
    log_level: str | None,
) -> None:
    from raggit.core.audit import log_event
    from raggit.db.repository import (
        ChunkRepository,
        DocumentRepository,
        EmbeddingCollectionRepository,
    )
    from raggit.db.vector import VectorStore
    from raggit.ingestion.embedder import create_embedder
    from raggit.llm.augmenter import augment_and_answer
    from raggit.llm.factory import create_llm
    from raggit.retrieval.engine import RetrievalEngine

    settings = get_settings()
    config = settings.rag_config
    _apply_overrides(
        config,
        top_k=top_k,
        min_top_k=min_top_k,
        max_top_k=max_top_k,
        top_k_ratio=top_k_ratio,
        rrf_k=rrf_k,
        parent_window=parent_window,
        min_score=min_score,
        rewrite=rewrite,
        multi_query_count=multi_query_count,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        reranker_top_n=reranker_top_n,
        refuse_on_empty=refuse_on_empty,
        refuse_on_low_score=refuse_on_low_score,
        min_answer_score=min_answer_score,
        groundedness_check=groundedness_check,
        pii_redaction=pii_redaction,
        prompt_injection_hardening=prompt_injection_hardening,
        log_level=log_level,
    )
    configure_logging(config.log_level)

    doc_uuids: list[UUID] = []
    if document_ids:
        for raw in document_ids:
            try:
                doc_uuids.append(UUID(raw))
            except ValueError as exc:
                console.print(f"[red]Invalid document id: {raw}[/red]")
                raise typer.Exit(1) from exc

    filters = QueryFilters(
        source_uri_prefix=source_prefix,
        filename_prefix=filename_prefix,
        tenant_id=tenant,
        tags=tags or [],
        document_ids=doc_uuids,
        created_after=created_after,
        created_before=created_before,
    )

    embedder = create_embedder(config.embedding)
    vector_store = VectorStore(config)

    async with AsyncSessionLocal() as session, session.begin():
        await log_event(
            session,
            level="INFO",
            component="raggit.cli.query",
            message="Query received",
            extra={
                "question": question,
                "filters": filters.model_dump(mode="json"),
                "rewrite": rewrite.value,
            },
        )

        active = await EmbeddingCollectionRepository(session).get_active()
        if active is not None:
            vector_store.set_collection(active.name)

        chunk_repo = ChunkRepository(session)
        doc_repo = DocumentRepository(session)

        llm = None
        llm_ready = config.llm.provider == "ollama" or bool(config.llm.api_key)
        if not no_llm and config.llm.provider and llm_ready:
            llm = create_llm(config.llm)

        engine = RetrievalEngine(
            embedder=embedder,
            vector_store=vector_store,
            chunk_repo=chunk_repo,
            document_repo=doc_repo,
            config=config,
            llm=llm,
        )

        with console.status("[bold green]Retrieving relevant chunks..."):
            result = await engine.retrieve(question, filters=filters)

        console.print(_format_retrieved_table(result))

        if result.refused and not llm:
            console.print(
                Panel(
                    f"[yellow]{result.refusal_reason}[/yellow]",
                    title="Refused",
                    border_style="yellow",
                )
            )
        elif llm is not None:
            with console.status("[bold green]Generating answer..."):
                result = await augment_and_answer(llm, result, safety=config.safety)

            answer_panel = Panel(
                result.answer or "[dim]No answer produced.[/dim]",
                title="Answer",
                border_style="cyan",
            )
            console.print(answer_panel)

            if result.grounded is False:
                console.print("[yellow]Groundedness check failed[/yellow]")

            if result.citations:
                console.print(_format_citation_tree(result.citations))

            await log_event(
                session,
                level="INFO",
                component="raggit.cli.query",
                message="Answer generated",
                extra={
                    "question": question,
                    "answer": result.answer,
                    "refused": result.refused,
                    "grounded": result.grounded,
                    "citation_count": len(result.citations),
                },
            )
        else:
            console.print("\n[yellow]No LLM configured; showing retrieved chunks only.[/yellow]")
            if result.citations:
                console.print(_format_citation_tree(result.citations))

        await engine.close()
