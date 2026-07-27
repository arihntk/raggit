"""raggit eval command."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from raggit.core.config import get_settings
from raggit.core.logging import configure_logging
from raggit.eval import (
    EvalRunner,
    load_dataset,
    render_console_report,
    save_dataset,
    save_report,
)
from raggit.eval.models import DEFAULT_METRICS, EvalDataset
from raggit.eval.reports import ReportFormatError

console = Console()


def _build_dataset(
    path: Path | None,
    *,
    name: str | None,
    description: str | None,
    metrics: list[str] | None,
    k_values: list[int] | None,
    generate: bool,
) -> EvalDataset:
    """Load or generate an evaluation dataset."""
    if path is not None:
        return load_dataset(str(path))
    if generate:
        dataset = EvalDataset(
            name=name or "generated-eval",
            description=description,
            metrics=list(metrics or DEFAULT_METRICS),
            k_values=list(k_values or [5, 10]),
            tests=[],
        )
        save_dataset(str(Path.cwd() / f"{dataset.name}.yaml"), dataset)
        console.print(
            f"[green]Generated empty evaluation dataset: {dataset.name}.yaml[/green]"
        )
        raise typer.Exit(0)
    msg = "Provide a dataset path or use --generate to create an empty dataset."
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


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
        """Run an evaluation dataset against the configured raggit system."""
        dataset = _build_dataset(
            path,
            name=name,
            description=description,
            metrics=metrics,
            k_values=k_values,
            generate=generate,
        )
        asyncio.run(_run_eval(dataset, output, output_format, log_level))


async def _run_eval(
    dataset: EvalDataset,
    output: Path | None,
    output_format: str | None,
    log_level: str | None,
) -> None:
    settings = get_settings()
    config = settings.rag_config
    if log_level is not None:
        config.log_level = log_level
    configure_logging(config.log_level)

    if not dataset.tests:
        console.print(
            Panel(
                "[yellow]Dataset contains no test cases.[/yellow]",
                title="raggit eval",
                border_style="yellow",
            )
        )
        raise typer.Exit(0)

    runner = EvalRunner(config)
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
