"""CSV/JSONL annotation loading with line-aware normalization.

Two ingestion paths share the exact same per-row normalization:

* the legacy in-memory path (``load_dataset``) decodes the whole file and
  returns a fully materialized :class:`Dataset`;
* the bounded-memory path (``csv_events`` / ``jsonl_events``) streams decoded
  physical lines from a fixed-size binary buffer and yields one
  :class:`ParseEvent` at a time so a temporary SQLite stage can spill to disk.

Both paths report the same physical line numbers, including newlines embedded
in quoted CSV fields, the same empty-label and malformed-row findings and the
same handling of GBK / UTF-8-SIG inputs.
"""

from __future__ import annotations

import codecs
import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import DatasetSpec
from .errors import DataError
from .hashing import sha256_file
from .models import Annotation, Dataset, ValidationFinding

REQUIRED_COLUMNS = ("annotator_id", "item_id", "label")

# Physical line terminators, matching the boundary set used by ``str.splitlines``
# so streamed JSONL line numbers equal the legacy ``text.splitlines()`` numbering.
# CRLF is listed first so the alternation prefers it when both bytes are present.
_TERMINATORS = ("\r\n", "\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e",
                "\x85", " ", " ")
_TERMINATOR_RE = re.compile("|".join(re.escape(seq) for seq in _TERMINATORS))


def _next_terminator(buffer: str) -> re.Match | None:
    """Find the earliest terminator, treating a trailing CR as non-final.

    A lone ``\\r`` at the very end of the available buffer is ignored because
    the next chunk may start with ``\\n``, in which case it is a single CRLF
    boundary rather than two boundaries.
    """
    position = 0
    while True:
        match = _TERMINATOR_RE.search(buffer, position)
        if match is None:
            return None
        if match.group() == "\r" and match.end() == len(buffer):
            return None
        return match

DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ParseEvent:
    """One normalized input line: either a valid annotation or a parse finding."""

    annotation: Annotation | None
    finding: ValidationFinding | None

    @property
    def line_no(self) -> int:
        target = self.annotation if self.annotation is not None else self.finding
        return target.line_no if target is not None else 0


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
            )
        ]
    if number < 0.0 or number > 1.0:
        return number, [
            ValidationFinding(
                "INVALID_CONFIDENCE",
                "warning",
                "confidence is outside [0, 1]",
                line_no=line_no,
            )
        ]
    return number, []


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


def _open_binary(path: Path):
    if not path.exists() or not path.is_file():
        raise DataError("DATA_FILE_NOT_FOUND", f"dataset not found: {path}", "dataset.path")
    try:
        return path.open("rb")
    except OSError as exc:
        raise DataError("DATA_READ_ERROR", f"cannot read dataset: {exc}", "dataset.path") from exc


def stream_text_lines(
    path: Path,
    encoding: str,
    hasher: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, physical_line)`` without materializing the file.

    The binary file is read in fixed-size chunks, incrementally decoded and
    split on every physical line terminator recognized by ``str.splitlines``.
    ``hasher`` is updated with each raw byte block so the caller obtains the
    same SHA-256 as a whole-file hash. Memory stays bounded by ``chunk_size``
    plus one partially-decoded line.
    """
    handle = _open_binary(path)
    decoder = codecs.getincrementaldecoder(encoding)()
    buffer = ""
    pending_cr = False
    line_no = 0
    try:
        while True:
            try:
                block = handle.read(chunk_size)
            except OSError as exc:
                raise DataError(
                    "DATA_READ_ERROR", f"cannot read dataset: {exc}", "dataset.path"
                ) from exc
            if not block:
                break
            hasher.update(block)
            try:
                decoded = decoder.decode(block)
            except UnicodeError as exc:
                raise DataError(
                    "DATA_ENCODING",
                    f"cannot decode dataset as {encoding}",
                    "dataset.encoding",
                ) from exc
            # Re-attach a CR held back at the previous chunk boundary; the new
            # bytes may start with LF, making CRLF a single boundary.
            if pending_cr:
                buffer += "\r" + decoded
                pending_cr = False
            else:
                buffer += decoded
            while True:
                match = _next_terminator(buffer)
                if match is None:
                    # Hold back a lone trailing CR until the next read or EOF
                    # so a CRLF split across chunks is not seen as two lines.
                    if buffer.endswith("\r"):
                        pending_cr = True
                        buffer = buffer[:-1]
                    break
                line_no += 1
                end = match.end()
                yield line_no, buffer[:end]
                buffer = buffer[end:]
        try:
            tail = decoder.decode(b"", final=True)
        except UnicodeError as exc:
            raise DataError(
                "DATA_ENCODING",
                f"cannot decode dataset as {encoding}",
                "dataset.encoding",
            ) from exc
        buffer += tail
        if pending_cr:
            buffer = "\r" + buffer
        if buffer:
            line_no += 1
            yield line_no, buffer
    finally:
        handle.close()


def _normalize_header(header: list[str]) -> list[str]:
    return [cell.strip().lower().lstrip("﻿") for cell in header]


def _check_header(header: list[str]) -> dict[str, int]:
    header = _normalize_header(header)
    if len(set(header)) != len(header):
        raise DataError("DATA_HEADER", "dataset header contains duplicates", "dataset.path")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise DataError(
            "DATA_HEADER", f"dataset missing column {missing[0]}", "dataset.path"
        )
    return {name: index for index, name in enumerate(header)}


def _csv_row_result(
    row: list[str],
    line_no: int,
    width: int,
    positions: dict[str, int],
    confidence_index: int | None,
) -> ParseEvent:
    """Normalize one CSV row; shared by the in-memory and streamed paths."""
    if not row or all(not cell.strip() for cell in row):
        return ParseEvent(None, None)
    if len(row) != width:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW",
                "error",
                f"expected {width} columns, found {len(row)}",
                line_no=line_no,
            ),
        )
    annotator_id = row[positions["annotator_id"]].strip()
    item_id = row[positions["item_id"]].strip()
    if not annotator_id or not item_id:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW",
                "error",
                "annotator_id and item_id must be non-empty",
                line_no=line_no,
            ),
        )
    try:
        labels = normalize_labels(row[positions["label"]])
    except ValueError:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW", "error", "label value is invalid", line_no=line_no
            ),
        )
    confidence = None
    findings: list[ValidationFinding] = []
    if confidence_index is not None:
        confidence, findings = _confidence(row[confidence_index], line_no)
    annotation = Annotation(annotator_id, item_id, labels, confidence, line_no)
    if findings:
        # A row carries at most one confidence finding; keep the event uniform.
        return ParseEvent(annotation, findings[0])
    return ParseEvent(annotation, None)


def csv_events(
    path: Path,
    encoding: str,
    hasher: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[ParseEvent]:
    """Stream a CSV file as normalized events, honoring quoted newlines."""
    reader = csv.reader(
        line for _, line in stream_text_lines(path, encoding, hasher, chunk_size)
    )
    try:
        header = next(reader)
    except StopIteration:
        return
    positions = _check_header(header)
    confidence_index = positions.get("confidence")
    width = len(header)
    for row in reader:
        event = _csv_row_result(
            row, reader.line_num, width, positions, confidence_index
        )
        if event.annotation is None and event.finding is None:
            continue
        yield event


def _jsonl_row_result(line: str, line_no: int) -> ParseEvent:
    """Normalize one JSONL line; shared by the in-memory and streamed paths."""
    if not line.strip():
        return ParseEvent(None, None)
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW",
                "error",
                f"invalid JSON: {exc.msg}",
                line_no=line_no,
            ),
        )
    if not isinstance(raw, dict):
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW", "error", "JSONL row must be an object", line_no=line_no
            ),
        )
    annotator_id = str(raw.get("annotator_id", "")).strip()
    item_id = str(raw.get("item_id", "")).strip()
    if not annotator_id or not item_id or "label" not in raw:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW",
                "error",
                "annotator_id, item_id and label are required",
                line_no=line_no,
            ),
        )
    try:
        labels = normalize_labels(raw.get("label"))
    except ValueError:
        return ParseEvent(
            None,
            ValidationFinding(
                "MALFORMED_ROW", "error", "label value is invalid", line_no=line_no
            ),
        )
    confidence, confidence_findings = _confidence(raw.get("confidence"), line_no)
    annotation = Annotation(annotator_id, item_id, labels, confidence, line_no)
    if confidence_findings:
        return ParseEvent(annotation, confidence_findings[0])
    return ParseEvent(annotation, None)


def jsonl_events(
    path: Path,
    encoding: str,
    hasher: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[ParseEvent]:
    """Stream a JSONL file as normalized events, one object per physical line."""
    for line_no, line in stream_text_lines(path, encoding, hasher, chunk_size):
        event = _jsonl_row_result(line, line_no)
        if event.annotation is None and event.finding is None:
            continue
        yield event


def _load_csv(text: str, source: str, digest: str) -> Dataset:
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return Dataset((), source, "csv", digest)
    positions = _check_header(header)
    confidence_index = positions.get("confidence")
    annotations: list[Annotation] = []
    findings: list[ValidationFinding] = []
    width = len(header)
    for row in reader:
        line_no = reader.line_num
        event = _csv_row_result(
            row, line_no, width, positions, confidence_index
        )
        if event.annotation is None and event.finding is None:
            continue
        if event.annotation is not None:
            annotations.append(event.annotation)
        if event.finding is not None:
            findings.append(event.finding)
    return Dataset(tuple(annotations), source, "csv", digest, tuple(findings))


def _load_jsonl(text: str, source: str, digest: str) -> Dataset:
    annotations: list[Annotation] = []
    findings: list[ValidationFinding] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        event = _jsonl_row_result(line, line_no)
        if event.annotation is None and event.finding is None:
            continue
        if event.annotation is not None:
            annotations.append(event.annotation)
        if event.finding is not None:
            findings.append(event.finding)
    return Dataset(tuple(annotations), source, "jsonl", digest, tuple(findings))


def load_dataset(spec: DatasetSpec) -> Dataset:
    text, digest = _read_text(spec.path, spec.encoding)
    if spec.format == "csv":
        return _load_csv(text, spec.raw_path, digest)
    if spec.format == "jsonl":
        return _load_jsonl(text, spec.raw_path, digest)
    raise DataError("DATA_FORMAT", f"unsupported format {spec.format}", "dataset.format")
