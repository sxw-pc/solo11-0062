"""Pre-computation quality checks. Errors block; warnings are explicitly promoted in strict mode."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable

from .config import Config
from .hashing import sha256_text, stable_json
from .models import Annotation, Dataset, ValidationFinding

SEVERITY_ORDER = {"error": 0, "warning": 1}

# Findings caused by one concrete input record; identity is anchored on it.
RECORD_SCOPED_CODES = {
    "EMPTY_LABEL",
    "INVALID_CONFIDENCE",
    "LABEL_OUT_OF_SPACE",
    "MULTI_LABEL_NOT_ENABLED",
    "DUPLICATE_ANNOTATION",
}


@dataclass(frozen=True)
class ValidationResult:
    findings: tuple[ValidationFinding, ...]
    error_count: int
    warning_count: int
    promoted_count: int

    @property
    def blocked(self) -> bool:
        return self.error_count > 0

    @property
    def status(self) -> str:
        if self.error_count:
            return "fail"
        if self.warning_count:
            return "warn"
        return "pass"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "promoted_count": self.promoted_count,
            "findings": [item.to_dict() for item in self.findings],
        }


def _sort_key(finding: ValidationFinding) -> tuple:
    return (
        SEVERITY_ORDER.get(finding.severity, 9),
        finding.code,
        finding.line_no or 0,
        finding.item_id or "",
        finding.annotator_id or "",
        finding.message,
    )


def normalized_row_dict(row: Annotation) -> dict:
    """Canonical form of one input record; the finding's data fingerprint."""
    return {
        "annotator_id": row.annotator_id,
        "item_id": row.item_id,
        "labels": list(row.labels),
        "confidence": row.confidence,
    }


def _identity_scope(finding: ValidationFinding) -> tuple:
    """Stable locator for a finding, independent of line numbers and messages."""
    code = finding.code
    if code in RECORD_SCOPED_CODES:
        return ("record", finding.annotator_id or "", finding.item_id or "")
    if code == "UNKNOWN_ANNOTATOR":
        return ("annotator", finding.annotator_id or "")
    if code == "INSUFFICIENT_ANNOTATORS":
        return ("item", finding.item_id or "")
    if finding.annotator_id and finding.item_id:
        return ("record", finding.annotator_id, finding.item_id)
    if finding.item_id:
        return ("item", finding.item_id)
    if finding.annotator_id:
        return ("annotator", finding.annotator_id)
    return ("message", finding.message)


def assign_identities(
    findings: Iterable[ValidationFinding],
    dataset: Dataset,
    persist_rows: bool = False,
) -> list[ValidationFinding]:
    """Attach cross-run identity material to every finding.

    ``finding_key`` identifies the same logical problem across runs: the code
    plus a line-number-free scope plus an occurrence ordinal that keeps
    multiple findings on one record (e.g. repeated duplicate keys) distinct.
    ``content_key`` fingerprints the line-free semantic content, so a value
    change (revised label, new count) is visible without breaking identity.
    ``row_digest`` fingerprints the normalized source record when the finding
    can be located to one. ``row`` keeps that record verbatim only when the
    plan opts into ``persist_raw_rows``.
    """
    findings = list(findings)
    rows_by_line = {row.line_no: row for row in dataset.annotations}
    scopes = {id(item): _identity_scope(item) for item in findings}
    groups: dict[tuple, list[ValidationFinding]] = {}
    for item in findings:
        groups.setdefault((item.code, scopes[id(item)]), []).append(item)
    ordinals: dict[int, int] = {}
    for members in groups.values():
        ordered = sorted(
            members,
            key=lambda item: (item.line_no or 0, item.message, item.content),
        )
        for index, item in enumerate(ordered, start=1):
            ordinals[id(item)] = index
    assigned: list[ValidationFinding] = []
    for item in findings:
        key_payload = [item.code, *scopes[id(item)], ordinals[id(item)]]
        content_payload = [item.code, item.severity, item.promoted, item.content]
        record = rows_by_line.get(item.line_no) if item.line_no is not None else None
        row_dict = normalized_row_dict(record) if record is not None else None
        assigned.append(
            dataclasses.replace(
                item,
                finding_key=sha256_text(stable_json(key_payload)),
                content_key=sha256_text(stable_json(content_payload)),
                row_digest=(
                    sha256_text(stable_json(row_dict)) if row_dict is not None else None
                ),
                row=row_dict if (persist_rows and row_dict is not None) else None,
            )
        )
    return assigned


def _duplicate_findings(annotations: Iterable[Annotation]) -> list[ValidationFinding]:
    grouped: dict[tuple[str, str], list[Annotation]] = {}
    for row in annotations:
        grouped.setdefault((row.annotator_id, row.item_id), []).append(row)
    findings: list[ValidationFinding] = []
    for (annotator_id, item_id), rows in sorted(grouped.items()):
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda row: row.line_no)
        first_line = ordered[0].line_no
        for duplicate in ordered[1:]:
            findings.append(
                ValidationFinding(
                    "DUPLICATE_ANNOTATION",
                    "error",
                    f"duplicate annotation for {annotator_id}/{item_id}; first at line {first_line}",
                    item_id=item_id,
                    annotator_id=annotator_id,
                    line_no=duplicate.line_no,
                )
            )
    return findings


def validate_dataset(dataset: Dataset, config: Config) -> ValidationResult:
    findings: list[ValidationFinding] = list(dataset.parse_findings)
    findings.extend(_duplicate_findings(dataset.annotations))
    if dataset.records == 0:
        findings.append(
            ValidationFinding("DATA_EMPTY", "error", "dataset contains no valid records")
        )

    allowed_labels = set(config.label_space)
    allowed_annotators = set(config.annotators)
    labels_by_item: dict[str, int] = {item_id: 0 for item_id in dataset.item_ids}
    unknown_seen: set[str] = set()

    for row in dataset.annotations:
        if row.is_missing:
            findings.append(
                ValidationFinding(
                    "EMPTY_LABEL",
                    "warning",
                    "empty label treated as a missing annotation",
                    item_id=row.item_id,
                    annotator_id=row.annotator_id,
                    line_no=row.line_no,
                )
            )
        else:
            labels_by_item[row.item_id] = labels_by_item.get(row.item_id, 0) + 1
        if config.annotators and row.annotator_id not in allowed_annotators:
            if row.annotator_id not in unknown_seen:
                unknown_seen.add(row.annotator_id)
                findings.append(
                    ValidationFinding(
                        "UNKNOWN_ANNOTATOR",
                        "error",
                        f"annotator {row.annotator_id} is not in the whitelist",
                        item_id=row.item_id,
                        annotator_id=row.annotator_id,
                        line_no=row.line_no,
                    )
                )
        for label in row.labels:
            if label not in allowed_labels:
                findings.append(
                    ValidationFinding(
                        "LABEL_OUT_OF_SPACE",
                        "error",
                        f"label {label} is outside label_space",
                        item_id=row.item_id,
                        annotator_id=row.annotator_id,
                        line_no=row.line_no,
                        content=f"label={label}",
                    )
                )
        if not config.multi_label and len(row.labels) > 1:
            findings.append(
                ValidationFinding(
                    "MULTI_LABEL_NOT_ENABLED",
                    "error",
                    "multiple labels supplied while multi_label is false",
                    item_id=row.item_id,
                    annotator_id=row.annotator_id,
                    line_no=row.line_no,
                    content="labels=" + ",".join(row.labels),
                )
            )

    for item_id, count in sorted(labels_by_item.items()):
        if count < config.min_annotators_per_item:
            findings.append(
                ValidationFinding(
                    "INSUFFICIENT_ANNOTATORS",
                    "warning",
                    f"item has {count} non-empty annotations; minimum is "
                    f"{config.min_annotators_per_item}",
                    item_id=item_id,
                    content=(
                        f"count={count};min={config.min_annotators_per_item}"
                    ),
                )
            )

    if config.strict:
        findings = [
            dataclasses.replace(
                item,
                severity="error",
                message=f"strict mode promotes warning: {item.message}",
                promoted=True,
            )
            if item.severity == "warning"
            else item
            for item in findings
        ]
    findings = assign_identities(
        findings, dataset, persist_rows=config.persist_raw_rows
    )
    findings.sort(key=_sort_key)
    error_count = sum(1 for item in findings if item.severity == "error")
    warning_count = sum(1 for item in findings if item.severity == "warning")
    promoted_count = sum(1 for item in findings if item.promoted)
    return ValidationResult(tuple(findings), error_count, warning_count, promoted_count)
