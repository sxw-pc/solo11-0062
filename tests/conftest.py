from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from tagconsensus.config import Config, load_config
from tagconsensus.models import Annotation, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["topic_alpha", "topic_beta", "topic_gamma", "topic_delta", "other"]


def make_dataset(rows: list[tuple[str, str, Any]], source: str = "synthetic") -> Dataset:
    annotations = []
    for index, (annotator, item, labels) in enumerate(rows, start=1):
        if isinstance(labels, str):
            normalized = () if not labels.strip() else (labels.strip().lower(),)
        else:
            normalized = tuple(sorted({str(value).strip().lower() for value in labels
                                       if str(value).strip()}))
        annotations.append(Annotation(annotator, item, normalized, None, index))
    return Dataset(tuple(annotations), source, "csv", "test-sha")


def make_config(
    tmp_path: Path,
    data_path: Path | None = None,
    *,
    metric: str = "pairwise_agreement",
    label_space: list[str] | None = None,
    annotators: list[str] | None = None,
    multi_label: bool = False,
    min_annotators: int = 2,
    review_threshold: float = 0.6,
    outlier_threshold: float = 0.65,
    strict: bool = False,
    adjudication_file: Path | None = None,
    db_path: Path | None = None,
    output_dir: Path | None = None,
    assignment_manifest: Path | str | dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    data_path = data_path or (tmp_path / "data.csv")
    if not data_path.exists():
        data_path.write_text("annotator_id,item_id,label\n", encoding="utf-8")
    payload: dict[str, Any] = {
        "dataset": {"path": str(data_path), "format": "csv", "encoding": "utf-8"},
        "label_space": label_space or list(LABELS),
        "annotators": annotators if annotators is not None else [],
        "metric": metric,
        "multi_label": multi_label,
        "min_annotators_per_item": min_annotators,
        "review_threshold": review_threshold,
        "outlier_threshold": outlier_threshold,
        "output_dir": str(output_dir or (tmp_path / "output")),
        "db_path": str(db_path or (tmp_path / "var" / "tagconsensus.db")),
        "strict": strict,
    }
    if adjudication_file is not None:
        payload["adjudication_file"] = str(adjudication_file)
    if assignment_manifest is not None:
        if isinstance(assignment_manifest, Path):
            payload["assignment_manifest"] = str(assignment_manifest)
        else:
            payload["assignment_manifest"] = assignment_manifest
    if extra:
        payload.update(extra)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def write_manifest_csv(path: Path, pairs: list[tuple[str, str]]) -> Path:
    lines = ["annotator_id,item_id"]
    lines.extend(f"{annotator_id},{item_id}" for annotator_id, item_id in pairs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def metric_config(tmp_path: Path, metric: str, **overrides: Any) -> Config:
    return load_config(make_config(tmp_path, metric=metric, **overrides))


def write_jsonl(path: Path, rows: list[dict[str, Any]], encoding: str = "utf-8") -> Path:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding=encoding,
    )
    return path


@pytest.fixture
def demo_rows() -> list[tuple[str, str, str]]:
    return [
        ("A01", "it_0001", "topic_alpha"),
        ("A02", "it_0001", "topic_alpha"),
        ("A03", "it_0001", "topic_alpha"),
        ("A01", "it_0002", "topic_beta"),
        ("A02", "it_0002", "topic_beta"),
        ("A03", "it_0002", "topic_alpha"),
    ]
