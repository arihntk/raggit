"""raggit eval command – three-tier evaluation.

Tiers:
- **component**: isolated primitives (parser, chunker, cleaner, PII, injection,
  sanitizer, embedder, RRF, reranker, safety, storage, watcher, retriever)
- **pipeline**: ingestion (parse→chunk→clean→embed) and retrieval
  (sanitize→rewrite→BM25/semantic→RRF→rerank→threshold→parent→traversal)
- **system**: end-to-end (ingestion + retrieval + LLM) – the original EvalRunner
- **all**: runs component + pipeline + system sequentially

Every feature of raggit maps to at least one metric; use
``raggit eval --list-metrics`` or ``--comprehensive`` to generate a
dataset that covers everything.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging
from raggit.eval import (
    EvalRunner,
    load_dataset,
    render_console_report,
    save_dataset,
    save_report,
)
from raggit.eval.loader import build_comprehensive_example, build_example_dataset
from raggit.eval.models import ALL_METRICS, ComponentType, EvalKind, PipelineType
from raggit.eval.reports import ReportFormatError

console = Console()


def _build_dataset(
    path: Path | None,
    *,
    name: str | None,
    description: str | None,
    metrics: list[str] | None,
    k_values: list[int] | None,
    kind: EvalKind | None,
    component: ComponentType | None,
    pipeline: PipelineType | None,
    comprehensive: bool,
    generate: bool,
) -> EvalDataset:  # type: ignore[name-defined]
    from raggit.eval.models import EvalDataset

    if comprehensive:
        dataset = build_comprehensive_example()
        if name:
            dataset.name = name
        if description:
            dataset.description = description
        save_dataset(str(Path.cwd() / f"{dataset.name}.yaml"), dataset)
        console.print(f"[green]Generated comprehensive system dataset: {dataset.name}.yaml[/green]")
        console.print(f"[dim]Covers {len(dataset.metrics)} metrics across all features[/dim]")
        raise typer.Exit(0)

    if path is not None:
        return load_dataset(str(path))

    if generate:
        # Use explicit kind if given, else infer from component/pipeline
        if kind is None:
            if component is not None:
                kind = EvalKind.COMPONENT
            elif pipeline is not None:
                kind = EvalKind.PIPELINE
            else:
                kind = EvalKind.SYSTEM
        if kind == EvalKind.ALL:
            from raggit.eval.loader import build_all_tiers_example

            dataset = build_all_tiers_example()
        else:
            dataset = build_example_dataset(kind=kind, component=component, pipeline=pipeline)
        if name:
            dataset.name = name
        if description:
            dataset.description = description
        if metrics:
            dataset.metrics = list(metrics)
        if k_values:
            dataset.k_values = list(k_values)
        # If single component requested but metrics not overridden, keep component's preset
        save_dataset(str(Path.cwd() / f"{dataset.name}.yaml"), dataset)
        console.print(f"[green]Generated {kind.value} evaluation dataset: {dataset.name}.yaml[/green]")
        if component:
            console.print(f"[dim]Component: {component.value}[/dim]")
        if pipeline:
            console.print(f"[dim]Pipeline: {pipeline.value}[/dim]")
        raise typer.Exit(0)

    msg = "Provide a dataset path, --comprehensive, or --generate to create a dataset."
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


def _list_metrics() -> None:
    table = Table(title="Available Evaluation Metrics (by tier)")
    table.add_column("Metric", style="cyan")
    table.add_column("Tier", style="magenta")
    table.add_column("Feature")
    # Component
    for m in [
        "parser_parse_success",
        "parser_text_fidelity",
        "parser_page_preservation",
        "chunker_section_preservation",
        "chunker_page_preservation",
        "chunker_function_boundary",
        "chunker_dedup_effectiveness",
        "cleaner_effectiveness",
        "pii_redaction_f1",
        "injection_hardening_recall",
        "sanitizer_keyword_recall",
        "rrf_fusion_quality",
        "reranker_gain",
        "safety_groundedness_accuracy",
        "embedder_cosine_accuracy",
        "storage_path_traversal_block_rate",
        "watcher_debounce_accuracy",
    ]:
        table.add_row(m, "component", "ingestion / retrieval primitive")
    for m in [
        "pipeline_ingestion_success_rate",
        "pipeline_retrieval_success_rate",
        "parent_window_gain",
        "traversal_precision",
    ]:
        table.add_row(m, "pipeline", "ingestion / retrieval chain")
    for m in [
        "retrieval_recall@k",
        "retrieval_precision@k",
        "retrieval_mrr",
        "retrieval_ndcg@k",
        "retrieval_hit_rate@k",
        "answer_exact_match",
        "answer_contains",
        "answer_semantic_similarity",
        "answer_llm_judge",
        "groundedness",
        "refusal_accuracy",
        "system_citation_precision",
        "system_hallucination_rate",
        "filter_tenant_accuracy",
        "latency_ms",
        "system_latency_p50",
        "system_latency_p95",
    ]:
        table.add_row(m, "system", "end-to-end / safety / filter")
    console.print(table)
    console.print(f"\n[dim]Total metrics: {len(ALL_METRICS)} – use --metric to filter[/dim]")
    raise typer.Exit(0)


def register_eval(app: typer.Typer) -> None:
    """Register the eval command with the CLI application."""

    @app.command()
    def eval(
        path: Path | None = typer.Argument(
            None,
            help="Path to an evaluation dataset (JSON or YAML).",
            exists=False,
        ),
        generate: bool = typer.Option(
            False, "--generate", help="Create an empty dataset YAML file and exit."
        ),
        comprehensive: bool = typer.Option(
            False, "--comprehensive", help="Generate a comprehensive dataset covering every feature."
        ),
        list_metrics: bool = typer.Option(
            False, "--list-metrics", help="List all available metrics and exit."
        ),
        kind: EvalKind | None = typer.Option(
            None, "--kind", help="Tier for generated dataset: component, pipeline, system, all."
        ),
        component: ComponentType | None = typer.Option(
            None, "--component", help="Component for --kind component (parser, chunker, etc.)."
        ),
        pipeline: PipelineType | None = typer.Option(
            None, "--pipeline", help="Pipeline for --kind pipeline (ingestion, retrieval, e2e)."
        ),
        name: str | None = typer.Option(
            None, "--name", help="Name for a generated dataset."
        ),
        description: str | None = typer.Option(
            None, "--description", help="Description for a generated dataset."
        ),
        metrics: list[str] = typer.Option(
            None,
            "--metric",
            help="Metrics to include when generating a dataset (repeatable).",
        ),
        k_values: list[int] = typer.Option(
            None,
            "--k",
            help="K values for @k metrics when generating a dataset (repeatable).",
        ),
        output: Path | None = typer.Option(
            None, "--output", help="Path to save the evaluation report (JSON or Markdown)."
        ),
        output_format: str | None = typer.Option(
            None,
            "--output-format",
            help="Override report format (json, markdown). Inferred from --output.",
        ),
        log_level: str | None = typer.Option(None, "--log-level", help="Override log level"),
    ) -> None:
        """Run an evaluation dataset against the configured raggit system.

        Three tiers are supported automatically based on the dataset's ``kind``:

        \b
        - component: isolated units (retriever, chunker, parser, ...)
        - pipeline: ingestion & retrieval chains
        - system: end-to-end with LLM (default)
        - all: run every tier

        Generate tier-specific templates with ``--generate``:

        \b
        raggit eval --generate --kind component --component retriever --name my-retriever
        raggit eval --generate --kind pipeline --pipeline ingestion
        raggit eval --comprehensive --name full-suite
        raggit eval --list-metrics
        """
        if list_metrics:
            _list_metrics()

        dataset = _build_dataset(
            path,
            name=name,
            description=description,
            metrics=metrics,
            k_values=k_values,
            kind=kind,
            component=component,
            pipeline=pipeline,
            comprehensive=comprehensive,
            generate=generate,
        )
        asyncio.run(_run_eval(dataset, output, output_format, log_level))


async def _run_eval(
    dataset,  # type: ignore[no-untyped-def]
    output: Path | None,
    output_format: str | None,
    log_level: str | None,
) -> None:
    from raggit.eval.models import EvalKind

    settings = get_settings()
    config = settings.rag_config
    if log_level is not None:
        config.log_level = log_level
    configure_logging(config.log_level)

    # Handle all-tiers: run each present tier sequentially and merge reports
    if dataset.kind == EvalKind.ALL:
        from raggit.eval.component import ComponentRunner
        from raggit.eval.pipeline import PipelineRunner
        from raggit.eval.system import SystemRunner

        # For --kind all without explicit tests, synthesize minimal suites
        # If dataset has no tests at all, generate example suites
        if not dataset.tests and not dataset.component_tests and not dataset.pipeline_tests:
            console.print("[yellow]Dataset kind is 'all' but has no tests – generating example suites[/yellow]")
            # Run a tiny system suite as representative
            from raggit.eval.loader import build_example_dataset

            dataset = build_example_dataset(kind=EvalKind.SYSTEM)

        # Dispatch based on what tests are present
        runners = []
        reports = []
        if dataset.component_tests or dataset.component is not None:
            cr = ComponentRunner(config)
            reports.append(await cr.run(dataset))
        if dataset.pipeline_tests or dataset.pipeline is not None:
            pr = PipelineRunner(config)
            rep = await pr.run(dataset)
            reports.append(rep)
            await pr.close()
        if dataset.tests:
            sr = SystemRunner(config)
            rep = await sr.run(dataset)
            reports.append(rep)
            await sr.close()
        # Merge: use first report as base, append aggregates
        if not reports:
            console.print("[yellow]No tests found for kind 'all'[/yellow]")
            raise typer.Exit(0)
        report = reports[0]
        for extra in reports[1:]:
            report.summary.aggregates.extend(extra.summary.aggregates)
            report.summary.per_test.extend(extra.summary.per_test)
            report.summary.total_tests += extra.summary.total_tests
            report.summary.passed_tests += extra.summary.passed_tests
            report.summary.failed_tests += extra.summary.failed_tests
        # Fall through to rendering
    elif dataset.kind == EvalKind.COMPONENT:
        from raggit.eval.component import ComponentRunner

        if not dataset.component_tests:
            console.print(
                Panel(
                    "[yellow]Component dataset has no component_tests.[/yellow]\n"
                    "Example: raggit eval --generate --kind component --component chunker",
                    title="raggit eval",
                    border_style="yellow",
                )
            )
            raise typer.Exit(0)
        runner = ComponentRunner(config)
        report = await runner.run(dataset)
    elif dataset.kind == EvalKind.PIPELINE:
        from raggit.eval.pipeline import PipelineRunner

        if not dataset.pipeline_tests:
            console.print(
                Panel(
                    "[yellow]Pipeline dataset has no pipeline_tests.[/yellow]\n"
                    "Example: raggit eval --generate --kind pipeline --pipeline retrieval",
                    title="raggit eval",
                    border_style="yellow",
                )
            )
            raise typer.Exit(0)
        runner = PipelineRunner(config)
        try:
            report = await runner.run(dataset)
        finally:
            await runner.close()
    else:  # SYSTEM and default
        if not dataset.tests:
            console.print(
                Panel(
                    "[yellow]Dataset contains no test cases.[/yellow]",
                    title="raggit eval",
                    border_style="yellow",
                )
            )
            raise typer.Exit(0)
        # Use SystemRunner for enriched system metrics (citation, p50/p95 etc.)
        from raggit.eval.system import SystemRunner

        runner = SystemRunner(config)
        try:
            report = await runner.run(dataset)
        finally:
            await runner.close()

    render_console_report(report, console=console)

    if output is not None:
        try:
            save_report(
                report,
                str(output.expanduser().resolve()),
                format=output_format,
            )
        except ReportFormatError as exc:
            console.print(f"[red]Failed to save report: {exc}[/red]")
            raise typer.Exit(1) from exc
        console.print(f"[green]Report saved to {output}[/green]")

    if report.summary.failed_tests:
        raise typer.Exit(1)
