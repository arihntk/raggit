"""Tests for raggit evaluation loader and models."""

from __future__ import annotations

from pathlib import Path

import pytest

from raggit.eval.loader import DatasetLoadError, load_dataset, save_dataset
from raggit.eval.models import DEFAULT_METRICS, EvalDataset, MetricName, TestCase


def test_default_metrics_contains_retrieval_and_answer() -> None:
    assert MetricName.RETRIEVAL_RECALL_AT_K in DEFAULT_METRICS
    assert MetricName.ANSWER_SEMANTIC_SIMILARITY in DEFAULT_METRICS


def test_eval_dataset_roundtrip(tmp_path: Path) -> None:
    dataset = EvalDataset(
        name="test-dataset",
        tests=[
            TestCase(
                id="t1",
                query="What is raggit?",
                expected_answer="A RAG system",
                tags=["basic"],
            )
        ],
    )
    path = tmp_path / "dataset.json"
    save_dataset(path, dataset)
    loaded = load_dataset(path)
    assert loaded.name == "test-dataset"
    assert len(loaded.tests) == 1
    assert loaded.tests[0].id == "t1"


def test_eval_dataset_yaml_roundtrip(tmp_path: Path) -> None:
    dataset = EvalDataset(
        name="yaml-dataset",
        tests=[
            TestCase(
                id="t1",
                query="q1",
                expected_chunk_ids=[],
            )
        ],
    )
    path = tmp_path / "dataset.yaml"
    save_dataset(path, dataset)
    loaded = load_dataset(path)
    assert loaded.name == "yaml-dataset"


def test_load_dataset_missing_file() -> None:
    with pytest.raises(DatasetLoadError):
        load_dataset("/nonexistent/path.json")


def test_load_dataset_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "dataset.txt"
    path.write_text("not yaml")
    with pytest.raises(DatasetLoadError):
        load_dataset(str(path))


def test_load_dataset_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(DatasetLoadError):
        load_dataset(str(path))
