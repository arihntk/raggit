"""Evaluation framework for raggit – three tiers."""

from __future__ import annotations

from raggit.eval.component import ComponentRunner
from raggit.eval.loader import DatasetLoadError, load_dataset, save_dataset
from raggit.eval.models import (
    ALL_METRICS,
    COMPONENT_METRICS,
    PIPELINE_METRICS,
    SYSTEM_METRICS,
    AnswerScores,
    ComponentScores,
    ComponentTestCase,
    ComponentType,
    DEFAULT_METRICS,
    EvalDataset,
    EvalKind,
    EvalReport,
    EvalSummary,
    MetricAggregate,
    MetricName,
    PipelineScores,
    PipelineTestCase,
    PipelineType,
    RetrievalScores,
    TestCase,
    TestResult,
)
from raggit.eval.pipeline import PipelineRunner
from raggit.eval.reports import (
    ReportFormatError,
    load_report,
    render_console_report,
    render_json_report,
    render_markdown_report,
    save_report,
)
from raggit.eval.runner import EvalRunner, EvaluationError
from raggit.eval.system import SystemRunner

__all__ = [
    "ALL_METRICS",
    "COMPONENT_METRICS",
    "PIPELINE_METRICS",
    "SYSTEM_METRICS",
    "AnswerScores",
    "ComponentRunner",
    "ComponentScores",
    "ComponentTestCase",
    "ComponentType",
    "DatasetLoadError",
    "EvalDataset",
    "EvalKind",
    "EvalReport",
    "EvalRunner",
    "EvalSummary",
    "EvaluationError",
    "MetricAggregate",
    "MetricName",
    "PipelineRunner",
    "PipelineScores",
    "PipelineTestCase",
    "PipelineType",
    "ReportFormatError",
    "RetrievalScores",
    "SystemRunner",
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
