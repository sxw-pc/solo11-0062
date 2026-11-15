"""Pre-computation quality checks. Errors block; warnings are explicitly promoted in strict mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .assignment import (
    AssignmentPlan,
    assignment_coverage,
    build_assignment_plan,
    coverage_findings,
)
from .config import Config
from .models import Annotation, Dataset, ValidationFinding

SEVERITY_ORDER = {"error": 0, "warning": 1}


@dataclass(frozen=True)
class ValidationResult:
    findings: tuple[ValidationFinding, ...]
    error_count: int
    warning_count: int
    promoted_count: int
    assignment_mode: str = "discovery"
    coverage: dict[str, Any] | None = None

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


def validate_dataset(
    dataset: Dataset, config: Config, plan: AssignmentPlan | None = None
) -> ValidationResult:
    if plan is None:
        plan = build_assignment_plan(config, dataset)
    findings: list[ValidationFinding] = list(dataset.parse_findings)
    findings.extend(plan.parse_findings)
    findings.extend(_duplicate_findings(dataset.annotations))
    if dataset.records == 0:
        findings.append(
            ValidationFinding("DATA_EMPTY", "error", "dataset contains no valid records")
        )
    findings.extend(coverage_findings(dataset, plan))

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
                )
            )

    if config.strict:
        findings = [
            ValidationFinding(
                item.code,
                "error",
                f"strict mode promotes warning: {item.message}",
                item.item_id,
                item.annotator_id,
                item.line_no,
                True,
            )
            if item.severity == "warning"
            else item
            for item in findings
        ]
    findings.sort(key=_sort_key)
    error_count = sum(1 for item in findings if item.severity == "error")
    warning_count = sum(1 for item in findings if item.severity == "warning")
    promoted_count = sum(1 for item in findings if item.promoted)
    return ValidationResult(
        tuple(findings),
        error_count,
        warning_count,
        promoted_count,
        plan.mode,
        assignment_coverage(dataset, plan),
    )
