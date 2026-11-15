"""CSV/JSONL parsing for the offline assignment manifest.

The manifest lists one expected ``annotator_id``/``item_id`` pair per line.
File-level problems (missing file, bad encoding, bad header) raise
``DataError``; per-line problems are returned as findings so they can be
reported through the normal validation report and ledger.
"""

from __future__ import annotations

import csv
import io
import json

from .errors import DataError
from .hashing import sha256_file
from .models import ValidationFinding

REQUIRED_COLUMNS = ("annotator_id", "item_id")
MANIFEST_FORMATS = ("csv", "jsonl")
MANIFEST_ENCODINGS = ("utf-8", "utf-8-sig", "gbk")
BOM = "﻿"


def read_manifest(spec) -> tuple[str, str]:
    if not spec.path.exists() or not spec.path.is_file():
        raise DataError(
            "ASSIGNMENT_FILE_NOT_FOUND",
            f"assignment manifest not found: {spec.path}",
            "assignment_manifest.path",
        )
    try:
        text = spec.path.read_bytes().decode(spec.encoding)
    except UnicodeError as exc:
        raise DataError(
            "ASSIGNMENT_ENCODING",
            f"cannot decode assignment manifest as {spec.encoding}",
            "assignment_manifest.encoding",
        ) from exc
    except OSError as exc:
        raise DataError(
            "ASSIGNMENT_READ_ERROR",
            f"cannot read assignment manifest: {exc}",
            "assignment_manifest.path",
        ) from exc
    return text, sha256_file(spec.path)


def _load_csv(text: str) -> tuple[list[tuple[str, str, int]], list[ValidationFinding]]:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return [], []
    header = [cell.strip().lower().lstrip(BOM) for cell in header]
    if len(set(header)) != len(header):
        raise DataError(
            "ASSIGNMENT_HEADER", "assignment manifest header contains duplicates",
            "assignment_manifest.path",
        )
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise DataError(
            "ASSIGNMENT_HEADER",
            f"assignment manifest missing column {missing[0]}",
            "assignment_manifest.path",
        )
    positions = {name: index for index, name in enumerate(header)}
    rows: list[tuple[str, str, int]] = []
    findings: list[ValidationFinding] = []
    for row in reader:
        line_no = reader.line_num
        if not row or all(not cell.strip() for cell in row):
            continue
        annotator_id = row[positions["annotator_id"]].strip()
        item_id = row[positions["item_id"]].strip()
        if not annotator_id or not item_id:
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_MALFORMED_ROW", "error",
                    "annotator_id and item_id must be non-empty", line_no=line_no,
                )
            )
            continue
        rows.append((annotator_id, item_id, line_no))
    return rows, findings


def _load_jsonl(text: str) -> tuple[list[tuple[str, str, int]], list[ValidationFinding]]:
    rows: list[tuple[str, str, int]] = []
    findings: list[ValidationFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_MALFORMED_ROW", "error",
                    f"invalid JSON: {exc.msg}", line_no=line_no,
                )
            )
            continue
        if not isinstance(raw, dict):
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_MALFORMED_ROW", "error",
                    "manifest row must be an object", line_no=line_no,
                )
            )
            continue
        annotator_id = str(raw.get("annotator_id", "")).strip()
        item_id = str(raw.get("item_id", "")).strip()
        if not annotator_id or not item_id:
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_MALFORMED_ROW", "error",
                    "annotator_id and item_id are required", line_no=line_no,
                )
            )
            continue
        rows.append((annotator_id, item_id, line_no))
    return rows, findings


def load_manifest_pairs(
    spec,
) -> tuple[list[tuple[str, str, int]], str, list[ValidationFinding]]:
    text, digest = read_manifest(spec)
    if spec.format == "csv":
        rows, findings = _load_csv(text)
    elif spec.format == "jsonl":
        rows, findings = _load_jsonl(text)
    else:  # pragma: no cover - guarded by config validation
        raise DataError(
            "ASSIGNMENT_FORMAT", f"unsupported manifest format {spec.format}",
            "assignment_manifest.format",
        )
    return rows, digest, findings
