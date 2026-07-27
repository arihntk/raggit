"""Tests for raggit evaluation reports."""

from __future__ import annotations

from pathlib import Path

import pytest

from raggit.eval.models import (
    AnswerScores,
    EvalDataset,
    EvalReport,
    EvalSummary,
    MetricAggregate,
    RetrievalScores,
    TestCase,
    TestResult,
)
from raggit.eval.reports import (
    ReportFormatError,
    load_report,
    render_json_report,
    render_markdown_report,
    save_report,
)


def _sample_report() -> EvalReport:
    result = TestResult(
        test_id="t1",
        query="q1",
        retrieval_scores=RetrievalScores(recall_at_k={"@5": 1.0}),
        answer_scores=AnswerScores(contains=True),
    )
    summary = EvalSummary(
        dataset_name="sample",
        dataset_version="1.0.0",
        total_tests=1,
        passed_tests=1,
        failed_tests=0,
        total_duration_ms=100.0,
        aggregates=[
            MetricAggregate(
                metric="retrieval_recall@5",
                mean=1.0,
                min=1.0,
                max=1.0,
                median=1.0,
                values=[1.0],
            )
        ],
        per_test=[result],
    )
    return EvalReport(
        summary=summary,
        dataset=EvalDataset(
            name="sample",
            tests=[TestCase(id="t1", query="q1")],
        ),
    )


def test_render_json_report() -> None:
    report = _sample_report()
    text = render_json_report(report)
    assert "sample" in text
    assert "retrieval_recall@5" in text


def test_render_markdown_report() -> None:
    report = _sample_report()
    text = render_markdown_report(report)
    assert "# Evaluation Report: sample" in text
    assert "retrieval_recall@5" in text


def test_save_report_json(tmp_path: Path) -> None:
    report = _sample_report()
    path = tmp_path / "report.json"
    save_report(report, path)
    loaded = load_report(path)
    assert loaded.summary.dataset_name == "sample"


def test_save_report_markdown(tmp_path: Path) -> None:
    report = _sample_report()
    path = tmp_path / "report.md"
    save_report(report, path)
    assert path.read_text().startswith("# Evaluation Report")


def test_save_report_unknown_format(tmp_path: Path) -> None:
    report = _sample_report()
    path = tmp_path / "report.txt"
    with pytest.raises(ReportFormatError):
        save_report(report, path)


def test_save_report_format_override(tmp_path: Path) -> None:
    report = _sample_report()
    path = tmp_path / "report.txt"
    save_report(report, path, format="json")
    assert path.exists()
    loaded = load_report(path)
    assert loaded.summary.dataset_name == "sample"
