"""Report generation for evaluation runs – rich terminal rendering.

This module renders a detailed, tier-aware report for all three eval kinds
(component / pipeline / system / all). The terminal output is designed to
be the primary way operators judge system health without opening JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from raggit.eval.models import EvalReport


class ReportFormatError(Exception):
    """Raised when an unsupported report format is requested."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _color_for_score(score: float | None) -> str:
    if score is None:
        return "dim"
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, bool):
        return "✓" if value else "✗"
    return str(value)


def _format_value(value: Any) -> Any:
    """Serialize metric values for JSON/Markdown output."""
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


# ---------------------------------------------------------------------------
# terminal rendering – detailed, tier-aware
# ---------------------------------------------------------------------------


def render_console_report(
    report: EvalReport,
    console: Console | None = None,
    golden_report: EvalReport | None = None,
) -> None:
    """Render a detailed human-readable evaluation report to the terminal.

    Args:
        report: The report to render.
        console: Optional Rich console (defaults to stdout).
        golden_report: Optional golden report to diff against (shows Δ).
    """
    if console is None:
        console = Console()

    summary = report.summary
    dataset = report.dataset

    # --- Header panel -------------------------------------------------
    kind_label = summary.kind.value.upper() if hasattr(summary.kind, "value") else str(summary.kind)
    tier_color = {"COMPONENT": "cyan", "PIPELINE": "magenta", "SYSTEM": "green", "ALL": "yellow"}.get(kind_label, "white")
    header_text = Text()
    header_text.append(f"{dataset.name} ", style="bold")
    header_text.append(f"v{dataset.version} ", style="dim")
    header_text.append(f"[{kind_label}]", style=f"bold {tier_color}")
    if dataset.description:
        header_text.append(f"\n{dataset.description}", style="dim")

    console.print(Panel(header_text, title="Evaluation Report", border_style=tier_color, box=box.ROUNDED))

    # --- Summary line -------------------------------------------------
    total = summary.total_tests
    passed = summary.passed_tests
    failed = summary.failed_tests
    pass_rate = (passed / total * 100) if total else 0
    pass_color = _color_for_score(pass_rate / 100)

    console.print(
        f"Tests: [bold]{total}[/] total • "
        f"[{pass_color}]{passed} passed ({pass_rate:.1f}%)[/] • "
        f"[red]{failed} failed[/] • "
        f"[dim]{summary.total_duration_ms:.1f} ms total[/] • "
        f"[dim]{dataset.kind.value if hasattr(dataset.kind, 'value') else dataset.kind} • k={dataset.k_values} • {len(dataset.metrics)} metrics[/]"
    )
    console.print(f"[dim]Config snapshot: {', '.join(k for k in report.config_snapshot.keys())[:120]}[/]" if report.config_snapshot else "")
    console.print()

    # --- Aggregate metrics table --------------------------------------
    if summary.aggregates:
        table = Table(title="Aggregate Metrics", box=box.SIMPLE_HEAD, show_lines=False)
        table.add_column("Metric", style="cyan", no_wrap=False)
        table.add_column("Mean", justify="right", style="bold")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Δ vs golden", justify="right", style="dim")

        # Build golden lookup for delta
        golden_map: dict[str, float] = {}
        if golden_report:
            for agg in golden_report.summary.aggregates:
                if agg.mean is not None:
                    golden_map[agg.metric] = agg.mean

        for agg in sorted(summary.aggregates, key=lambda a: a.metric):
            color = _color_for_score(agg.mean)
            mean_s = f"[{color}]{_fmt(agg.mean)}[/]" if agg.mean is not None else "-"
            # Delta vs golden
            delta_s = "-"
            if agg.metric in golden_map and agg.mean is not None:
                delta = agg.mean - golden_map[agg.metric]
                delta_color = "green" if delta >= 0 else "red"
                sign = "+" if delta >= 0 else ""
                delta_s = f"[{delta_color}]{sign}{delta:.4f}[/]"
            table.add_row(
                agg.metric,
                mean_s,
                _fmt(agg.min),
                _fmt(agg.max),
                _fmt(agg.median),
                delta_s,
            )
        console.print(table)
        console.print()

    # --- Per-test breakdown -------------------------------------------
    if summary.per_test:
        # Decide which columns to show based on tier
        is_component = summary.kind.value == "component" if hasattr(summary.kind, "value") else False
        is_pipeline = summary.kind.value == "pipeline" if hasattr(summary.kind, "value") else False

        table = Table(title="Per-Test Breakdown", box=box.MINIMAL, show_lines=False)
        table.add_column("#", style="dim", width=3)
        table.add_column("Test ID", style="bold cyan", no_wrap=False)
        table.add_column("Tier/Comp", style="magenta")
        table.add_column("Latency", justify="right")
        table.add_column("Key Metric", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Tags", style="dim")

        for idx, result in enumerate(summary.per_test, start=1):
            # Determine key metric to display
            key_metric = "-"
            key_val: float | None = None
            if result.component_scores and result.component_scores.metrics:
                # show first component metric
                first_k = next(iter(result.component_scores.metrics))
                key_val = result.component_scores.metrics[first_k]
                key_metric = f"{first_k.split('_')[-1]} {_fmt(key_val)}"
            elif result.pipeline_scores:
                if result.pipeline_scores.ingestion_success is not None:
                    key_val = result.pipeline_scores.ingestion_success
                    key_metric = f"ingest {_fmt(key_val)}"
                elif result.pipeline_scores.retrieval_success is not None:
                    key_val = result.pipeline_scores.retrieval_success
                    key_metric = f"retrieval {_fmt(key_val)}"
            elif result.retrieval_scores and result.retrieval_scores.recall_at_k:
                # take @5 recall
                v = result.retrieval_scores.recall_at_k.get("@5") or next(iter(result.retrieval_scores.recall_at_k.values()), None)
                key_val = v
                key_metric = f"recall@5 {_fmt(v)}"
            elif result.answer_scores and result.answer_scores.semantic_similarity is not None:
                key_val = result.answer_scores.semantic_similarity
                key_metric = f"sem {_fmt(key_val)}"
            elif "retrieval_recall@5" in result.metric_values:
                key_val = result.metric_values["retrieval_recall@5"]
                key_metric = f"recall@5 {_fmt(key_val)}"

            status = "[green]PASS[/]" if not result.errors else "[red]FAIL[/]"
            if result.refused and not result.errors:
                # Check if refusal was expected
                if result.refusal_accuracy == 1.0:
                    status = "[green]PASS[/] [dim](refused)[/]"
                else:
                    status = "[yellow]REFUSED[/]"

            tier_comp = result.component or result.pipeline or summary.kind.value[:4]
            latency_s = f"{result.latency_ms:.1f} ms" if result.latency_ms else "-"
            # Truncate long test IDs
            test_id_display = result.test_id if len(result.test_id) < 28 else result.test_id[:25] + "..."
            tags_s = ", ".join(result.tags[:2]) if result.tags else "-"

            # Color key metric by value
            if key_val is not None:
                km_color = _color_for_score(key_val if key_val <= 1 else key_val / 100)
                key_metric = f"[{km_color}]{key_metric}[/]"

            table.add_row(
                str(idx),
                test_id_display,
                tier_comp or "-",
                latency_s,
                key_metric,
                status,
                tags_s,
            )
        console.print(table)
        console.print()

        # --- Detailed per-test cards for failures or verbose ---------------------------------
        has_failures = any(r.errors for r in summary.per_test)
        if has_failures:
            console.print(Panel("[bold red]Failed tests – details[/]", border_style="red", box=box.ROUNDED))
            for result in summary.per_test:
                if result.errors:
                    err_text = "\n".join(f"• {e}" for e in result.errors)
                    details = Text()
                    details.append(f"{result.test_id}", style="bold red")
                    details.append(f"  query: {result.query[:80]}\n", style="dim")
                    details.append(f"errors: {err_text}\n", style="red")
                    if result.retrieval_scores and result.retrieval_scores.recall_at_k:
                        details.append(f"retrieval: {result.retrieval_scores.recall_at_k}\n", style="dim")
                    if result.answer is not None:
                        ans = result.answer[:200].replace("\n", " ")
                        details.append(f"answer: {ans}\n", style="dim")
                    console.print(Panel(details, border_style="red", box=box.SIMPLE))

        # --- System-tier answer & citation preview (first 3) --------------------------------
        system_examples = [r for r in summary.per_test if r.answer is not None][:3]
        if system_examples:
            console.print(Panel("[bold]Answer & Citation Preview (first 3 system tests)[/]", border_style="blue", box=box.ROUNDED))
            for r in system_examples:
                ans = (r.answer or "")[:300].replace("\n", " ")
                cites = len(r.retrieved_chunk_ids)
                grounded = r.answer_scores.groundedness
                grounded_s = "[green]grounded[/]" if grounded else "[red]ungrounded[/]" if grounded is False else "[dim]n/a[/]"
                console.print(f"[cyan]{r.test_id}[/] [dim]({r.latency_ms:.1f} ms, {cites} cites, {grounded_s})[/]")
                console.print(f"  [dim]Q:[/] {r.query[:100]}")
                console.print(f"  [dim]A:[/] {ans}{'...' if len(r.answer or '') > 300 else ''}")
                if r.answer_scores and r.answer_scores.llm_judge_reasoning:
                    console.print(f"  [dim]Judge:[/] {r.answer_scores.llm_judge_reasoning[:120]}")
                console.print()

        # --- Component details (first few) ----------------------------------------------------
        comp_examples = [r for r in summary.per_test if r.component_scores][:2]
        if comp_examples:
            console.print(Panel("[bold]Component Details (sample)[/]", border_style="cyan", box=box.ROUNDED))
            for r in comp_examples:
                console.print(f"[cyan]{r.test_id}[/] [magenta]{r.component}[/] [dim]{r.tags}[/]")
                for k, v in (r.component_scores.metrics if r.component_scores else {}).items():
                    col = _color_for_score(v)
                    console.print(f"  [{col}]{k}: {_fmt(v)}[/]")
                if r.component_scores and r.component_scores.details:
                    # show truncated details
                    for dk, dv in list(r.component_scores.details.items())[:3]:
                        console.print(f"  [dim]{dk}: {str(dv)[:80]}[/]")
                console.print()

    # --- Verdict -------------------------------------------------------
    if failed == 0 and total > 0:
        console.print(Panel("[bold green]✓ All tests passed – system is healthy[/]", border_style="green", box=box.HEAVY))
    elif failed > 0:
        console.print(Panel(f"[bold red]✗ {failed}/{total} tests failed – needs attention[/]", border_style="red", box=box.HEAVY))
    else:
        console.print(Panel("[yellow]No tests to evaluate[/]", border_style="yellow"))

    # --- Config snapshot footer ----------------------------------------
    if dataset.kind.value == "all" if hasattr(dataset.kind, "value") else False:
        console.print(f"\n[dim]Dataset kind: all (component={len(dataset.component_tests)}, pipeline={len(dataset.pipeline_tests)}, system={len(dataset.tests)})[/]")
    else:
        console.print(f"\n[dim]Dataset kind: {dataset.kind.value if hasattr(dataset.kind, 'value') else dataset.kind} • metrics: {', '.join(dataset.metrics[:5])}{'...' if len(dataset.metrics) > 5 else ''}[/]")


def _format_value(value: Any) -> Any:
    """Serialize metric values for JSON output."""
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, 6)
    return value


def render_json_report(report: EvalReport) -> str:
    """Return the report as a JSON string."""
    return report.model_dump_json(indent=2)


def render_markdown_report(report: EvalReport) -> str:
    """Return the report as a Markdown string."""
    summary = report.summary
    dataset = report.dataset
    lines: list[str] = [
        f"# Evaluation Report: {summary.dataset_name}",
        "",
        f"- **Version:** {summary.dataset_version}",
        f"- **Kind:** {summary.kind.value if hasattr(summary.kind, 'value') else summary.kind}",
        f"- **Total tests:** {summary.total_tests}",
        f"- **Passed:** {summary.passed_tests}",
        f"- **Failed:** {summary.failed_tests}",
        f"- **Pass rate:** {summary.passed_tests / summary.total_tests * 100:.1f}%" if summary.total_tests else "- **Pass rate:** n/a",
        f"- **Total duration:** {summary.total_duration_ms:.2f} ms",
        f"- **k values:** {dataset.k_values}",
        f"- **Metrics:** {', '.join(dataset.metrics)}",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Min | Max | Median |",
        "| --- | --- | --- | --- | --- |",
    ]

    for agg in sorted(summary.aggregates, key=lambda a: a.metric):
        lines.append(
            f"| {agg.metric} | "
            f"{_format_value(agg.mean)} | "
            f"{_format_value(agg.min)} | "
            f"{_format_value(agg.max)} | "
            f"{_format_value(agg.median)} |"
        )

    if summary.per_test:
        lines.extend(["", "## Per-Test Results", "", "| Test | Kind | Latency | Key Metric | Status | Tags |", "| --- | --- | --- | --- | --- | --- |"])
        for r in summary.per_test:
            key_metric = "-"
            if r.component_scores and r.component_scores.metrics:
                k = next(iter(r.component_scores.metrics))
                key_metric = f"{k}={_format_value(r.component_scores.metrics[k])}"
            elif r.retrieval_scores.recall_at_k:
                v = r.retrieval_scores.recall_at_k.get("@5", "-")
                key_metric = f"recall@5={_format_value(v)}"
            status = "PASS" if not r.errors else "FAIL"
            tier = r.component or r.pipeline or (summary.kind.value if hasattr(summary.kind, 'value') else "")
            lines.append(f"| {r.test_id} | {tier} | {_format_value(r.latency_ms)} ms | {key_metric} | {status} | {', '.join(r.tags)} |")

    if summary.failed_tests:
        lines.extend(["", "## Failed Tests", ""])
        for result in summary.per_test:
            if result.errors:
                lines.append(f"- **{result.test_id}:** {', '.join(result.errors)}")
                if result.query:
                    lines.append(f"  - Query: `{result.query[:80]}`")
                if result.answer:
                    lines.append(f"  - Answer: {result.answer[:120]}")

    # Add per-tier details
    lines.extend(["", "## Tier Details", ""])
    lines.append(f"- **Kind:** {dataset.kind.value if hasattr(dataset.kind, 'value') else dataset.kind}")
    if dataset.component:
        lines.append(f"- **Component:** {dataset.component}")
    if dataset.pipeline:
        lines.append(f"- **Pipeline:** {dataset.pipeline}")
    lines.append(f"- **Component tests:** {len(dataset.component_tests)}")
    lines.append(f"- **Pipeline tests:** {len(dataset.pipeline_tests)}")
    lines.append(f"- **System tests:** {len(dataset.tests)}")

    return "\n".join(lines)


def save_report(
    report: EvalReport,
    path: str | Path,
    format: str | None = None,
) -> None:
    """Save a report to disk.

    Args:
        report: Evaluation report to save.
        path: Destination path.
        format: Optional override format (json, md, markdown). Inferred from path.

    Raises:
        ReportFormatError: If the format cannot be inferred or is unsupported.
    """
    file_path = Path(path).expanduser().resolve()
    if format is None:
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            format = "json"
        elif suffix in {".md", ".markdown"}:
            format = "markdown"
        else:
            msg = f"Cannot infer report format from path: {file_path}"
            raise ReportFormatError(msg)

    file_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        file_path.write_text(render_json_report(report), encoding="utf-8")
    elif format in {"markdown", "md"}:
        file_path.write_text(render_markdown_report(report), encoding="utf-8")
    else:
        msg = f"Unsupported report format: {format}"
        raise ReportFormatError(msg)


def load_report(path: str | Path) -> EvalReport:
    """Load an evaluation report from a JSON file."""
    file_path = Path(path).expanduser().resolve()
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    return EvalReport(**raw)
