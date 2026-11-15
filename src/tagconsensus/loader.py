"""CSV/JSONL annotation loading with line-aware normalization."""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from pathlib import Path
from typing import Any

from .config import DatasetSpec
from .errors import DataError
from .hashing import sha256_file
from .models import Annotation, Dataset, ValidationFinding

REQUIRED_COLUMNS = ("annotator_id", "item_id", "label")


def normalize_labels(value: str | list[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    raw_values = value if isinstance(value, list) else [value]
    normalized: list[str] = []
    for item in raw_values:
        if not isinstance(item, str):
            raise ValueError("label must be a string or a list of strings")
        text = item.strip().lower()
        if text:
            normalized.append(text)
    return tuple(sorted(set(normalized)))


def _confidence(value: Any, line_no: int) -> tuple[float | None, list[ValidationFinding]]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None, []
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, [
            ValidationFinding(
                "INVALID_CONFIDENCE",
                "warning",
                "confidence is not numeric; treated as missing",
                line_no=line_no,
                content="not_numeric",
            )
        ]
    if number < 0.0 or number > 1.0:
        return number, [
            ValidationFinding(
                "INVALID_CONFIDENCE",
                "warning",
                "confidence is outside [0, 1]",
                line_no=line_no,
                content="out_of_range",
            )
        ]
    return number, []


def _locate(
    findings: list[ValidationFinding], annotator_id: str, item_id: str
) -> list[ValidationFinding]:
    """Anchor parse-time findings to their record so identity can locate them."""
    return [
        dataclasses.replace(item, annotator_id=annotator_id, item_id=item_id)
        for item in findings
    ]


def _read_text(path: Path, encoding: str) -> tuple[str, str]:
    if not path.exists() or not path.is_file():
        raise DataError("DATA_FILE_NOT_FOUND", f"dataset not found: {path}", "dataset.path")
    try:
        payload = path.read_bytes()
        text = payload.decode(encoding)
    except UnicodeError as exc:
        raise DataError(
            "DATA_ENCODING", f"cannot decode dataset as {encoding}", "dataset.encoding"
        ) from exc
    except OSError as exc:
        raise DataError("DATA_READ_ERROR", f"cannot read dataset: {exc}", "dataset.path") from exc
    return text, sha256_file(path)


def _load_csv(text: str, source: str, digest: str) -> Dataset:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return Dataset((), source, "csv", digest)
    header = [cell.strip().lower().lstrip("﻿") for cell in header]
    if len(set(header)) != len(header):
        raise DataError("DATA_HEADER", "dataset header contains duplicates", "dataset.path")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise DataError(
            "DATA_HEADER", f"dataset missing column {missing[0]}", "dataset.path"
        )
    positions = {name: index for index, name in enumerate(header)}
    confidence_index = positions.get("confidence")
    annotations: list[Annotation] = []
    findings: list[ValidationFinding] = []
    width = len(header)
    for row in reader:
        line_no = reader.line_num
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != width:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW",
                    "error",
                    f"expected {width} columns, found {len(row)}",
                    line_no=line_no,
                )
            )
            continue
        annotator_id = row[positions["annotator_id"]].strip()
        item_id = row[positions["item_id"]].strip()
        if not annotator_id or not item_id:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW",
                    "error",
                    "annotator_id and item_id must be non-empty",
                    line_no=line_no,
                )
            )
            continue
        try:
            labels = normalize_labels(row[positions["label"]])
        except ValueError:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW", "error", "label value is invalid", line_no=line_no
                )
            )
            continue
        confidence = None
        if confidence_index is not None:
            confidence, confidence_findings = _confidence(row[confidence_index], line_no)
            findings.extend(_locate(confidence_findings, annotator_id, item_id))
        annotations.append(
            Annotation(annotator_id, item_id, labels, confidence, line_no)
        )
    return Dataset(tuple(annotations), source, "csv", digest, tuple(findings))


def _load_jsonl(text: str, source: str, digest: str) -> Dataset:
    annotations: list[Annotation] = []
    findings: list[ValidationFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW",
                    "error",
                    f"invalid JSON: {exc.msg}",
                    line_no=line_no,
                )
            )
            continue
        if not isinstance(raw, dict):
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW", "error", "JSONL row must be an object", line_no=line_no
                )
            )
            continue
        annotator_id = str(raw.get("annotator_id", "")).strip()
        item_id = str(raw.get("item_id", "")).strip()
        if not annotator_id or not item_id or "label" not in raw:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW",
                    "error",
                    "annotator_id, item_id and label are required",
                    line_no=line_no,
                )
            )
            continue
        try:
            labels = normalize_labels(raw.get("label"))
        except ValueError:
            findings.append(
                ValidationFinding(
                    "MALFORMED_ROW", "error", "label value is invalid", line_no=line_no
                )
            )
            continue
        confidence, confidence_findings = _confidence(raw.get("confidence"), line_no)
        findings.extend(_locate(confidence_findings, annotator_id, item_id))
        annotations.append(
            Annotation(annotator_id, item_id, labels, confidence, line_no)
        )
    return Dataset(tuple(annotations), source, "jsonl", digest, tuple(findings))


def load_dataset(spec: DatasetSpec) -> Dataset:
    text, digest = _read_text(spec.path, spec.encoding)
    if spec.format == "csv":
        return _load_csv(text, spec.raw_path, digest)
    if spec.format == "jsonl":
        return _load_jsonl(text, spec.raw_path, digest)
    raise DataError("DATA_FORMAT", f"unsupported format {spec.format}", "dataset.format")
