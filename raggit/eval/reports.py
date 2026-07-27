"""Report generation for evaluation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from raggit.eval.models import EvalReport


class ReportFormatError(Exception):
    """Raised when an unsupported report format is requested."""


def render_console_report(report: EvalReport, console: Console | None = None) -> None:
    """Render a human-readable evaluation report to the console."""
    if console is None:
        console = Console()

    summary = report.summary
    console.print(
        f"[bold]Evaluation Report:[/bold] {summary.dataset_name} "
        f"v{summary.dataset_version}"
    )
    console.print(
        f"Tests: {summary.total_tests} total, "
        f"[green]{summary.passed_tests} passed[/green], "
        f"[red]{summary.failed_tests} failed[/red]"
    )
    console.print(f"Total duration: {summary.total_duration_ms:.2f} ms")
    console.print()

    if summary.aggregates:
        table = Table(title="Aggregate Metrics")
        table.add_column("Metric")
        table.add_column("Mean", justify="right")
        table.add_column("Min", justify="right")
        table.add_column("Max", justify="right")
        table.add_column("Median", justify="right")
        for agg in summary.aggregates:
            table.add_row(
                agg.metric,
                f"{agg.mean:.4f}" if agg.mean is not None else "-",
                f"{agg.min:.4f}" if agg.min is not None else "-",
                f"{agg.max:.4f}" if agg.max is not None else "-",
                f"{agg.median:.4f}" if agg.median is not None else "-",
            )
        console.print(table)

    if summary.failed_tests:
        console.print()
        console.print("[bold red]Failed tests:[/bold red]")
        for result in summary.per_test:
            if result.errors:
                console.print(f"  [red]•[/red] {result.test_id}: {', '.join(result.errors)}")


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
    lines: list[str] = [
        f"# Evaluation Report: {summary.dataset_name}",
        "",
        f"- **Version:** {summary.dataset_version}",
        f"- **Total tests:** {summary.total_tests}",
        f"- **Passed:** {summary.passed_tests}",
        f"- **Failed:** {summary.failed_tests}",
        f"- **Total duration:** {summary.total_duration_ms:.2f} ms",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | Min | Max | Median |",
        "| --- | --- | --- | --- | --- |",
    ]

    for agg in summary.aggregates:
        lines.append(
            f"| {agg.metric} | "
            f"{_format_value(agg.mean)} | "
            f"{_format_value(agg.min)} | "
            f"{_format_value(agg.max)} | "
            f"{_format_value(agg.median)} |"
        )

    if summary.failed_tests:
        lines.extend(["", "## Failed Tests", ""])
        for result in summary.per_test:
            if result.errors:
                lines.append(f"- **{result.test_id}:** {', '.join(result.errors)}")

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
