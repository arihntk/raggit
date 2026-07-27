"""Dataset loading utilities for evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from raggit.eval.models import EvalDataset, TestCase


class DatasetLoadError(Exception):
    """Raised when a dataset cannot be loaded or parsed."""


SUPPORTED_SUFFIXES = {".json", ".yaml", ".yml"}


def _coerce_test_case(data: dict[str, Any]) -> TestCase:
    """Build a TestCase from a raw dictionary, handling optional fields."""
    return TestCase(**data)


def load_dataset(path: str | Path) -> EvalDataset:
    """Load an evaluation dataset from a JSON or YAML file.

    Args:
        path: Path to the dataset file.

    Returns:
        Parsed EvalDataset.

    Raises:
        DatasetLoadError: If the file format is unsupported or parsing fails.
    """
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        msg = f"Dataset file not found: {file_path}"
        raise DatasetLoadError(msg)

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        msg = f"Unsupported dataset format: {suffix}. Use {SUPPORTED_SUFFIXES}."
        raise DatasetLoadError(msg)

    try:
        if suffix == ".json":
            raw = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        msg = f"Failed to parse dataset {file_path}: {exc}"
        raise DatasetLoadError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"Dataset root must be an object, got {type(raw).__name__}"
        raise DatasetLoadError(msg)

    try:
        return EvalDataset(**raw)
    except Exception as exc:
        msg = f"Failed to validate dataset {file_path}: {exc}"
        raise DatasetLoadError(msg) from exc


def save_dataset(path: str | Path, dataset: EvalDataset) -> None:
    """Save an evaluation dataset to JSON or YAML.

    Args:
        path: Destination path.
        dataset: Dataset to serialize.

    Raises:
        DatasetLoadError: If the file format is unsupported.
    """
    file_path = Path(path).expanduser().resolve()
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        file_path.write_text(
            dataset.model_dump_json(indent=2), encoding="utf-8"
        )
    elif suffix in {".yaml", ".yml"}:
        file_path.write_text(
            yaml.safe_dump(dataset.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
    else:
        msg = f"Unsupported dataset format: {suffix}"
        raise DatasetLoadError(msg)


def build_example_dataset() -> EvalDataset:
    """Return a minimal example dataset for documentation and bootstrapping."""
    return EvalDataset(
        name="example-eval",
        description="Example evaluation dataset showing the expected format.",
        metrics=["retrieval_recall@k", "retrieval_mrr", "answer_contains"],
        k_values=[5],
        tests=[
            TestCase(
                id="example-1",
                query="What does the example document describe?",
                expected_answer="This is an example document.",
                tags=["example"],
            )
        ],
    )
