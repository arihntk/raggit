"""Evaluation framework for raggit."""

from __future__ import annotations

from raggit.eval.loader import DatasetLoadError, load_dataset, save_dataset
from raggit.eval.models import (
    DEFAULT_METRICS,
    AnswerScores,
    EvalDataset,
    EvalReport,
    EvalSummary,
    MetricAggregate,
    MetricName,
    RetrievalScores,
    TestCase,
    TestResult,
)
from raggit.eval.reports import (
    ReportFormatError,
    load_report,
    render_console_report,
    render_json_report,
    render_markdown_report,
    save_report,
)
from raggit.eval.runner import EvalRunner, EvaluationError

__all__ = [
    "DEFAULT_METRICS",
    "AnswerScores",
    "DatasetLoadError",
    "EvalDataset",
    "EvalReport",
    "EvalRunner",
    "EvalSummary",
    "EvaluationError",
    "MetricAggregate",
    "MetricName",
    "ReportFormatError",
    "RetrievalScores",
    "TestCase",
    "TestResult",
    "load_dataset",
    "load_report",
    "render_console_report",
    "render_json_report",
    "render_markdown_report",
    "save_dataset",
    "save_report",
]
