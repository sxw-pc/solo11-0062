"""Offline assignment manifest: expected annotator/item pairs and coverage audit.

Two explicit allocation modes are supported:

* ``manifest`` – an offline manifest is the authority for who labels what;
* ``full`` – no manifest, but an explicit ``annotators`` whitelist is configured,
  so the expected set is derived as whitelist x the items present in the batch;
* ``discovery`` – neither is available; the roster is discovered from the data
  and no absent combination may be guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import ConfigError
from .manifest import load_manifest_pairs
from .models import Dataset, ValidationFinding

if TYPE_CHECKING:
    from .config import Config

ASSIGNMENT_MODES = ("manifest", "full", "discovery")
# Finding codes produced by coverage reconciliation, in severity order.
COVERAGE_FINDING_CODES = (
    "ASSIGNMENT_MISSING",
    "UNASSIGNED_ANNOTATION",
    "ASSIGNMENT_DUPLICATE_PAIR",
    "ASSIGNMENT_UNKNOWN_ANNOTATOR",
    "ASSIGNMENT_MALFORMED_ROW",
)


@dataclass(frozen=True)
class AssignmentManifestSpec:
    path: Path
    raw_path: str
    format: str
    encoding: str


@dataclass(frozen=True)
class AssignmentPlan:
    mode: str
    pairs: frozenset[tuple[str, str]]
    parse_findings: tuple[ValidationFinding, ...]
    manifest_path: str | None = None
    manifest_sha256: str | None = None


def parse_assignment_spec(raw: Any, base: Path) -> AssignmentManifestSpec:
    if isinstance(raw, str):
        raw = {"path": raw}
    if not isinstance(raw, dict):
        raise ConfigError(
            "CFG_TYPE_INVALID",
            "assignment_manifest must be a path string or a mapping",
            "assignment_manifest",
        )
    unknown = sorted(set(raw) - {"path", "format", "encoding"})
    if unknown:
        raise ConfigError(
            "CFG_UNKNOWN_FIELD",
            f"unknown assignment_manifest field {unknown[0]}",
            f"assignment_manifest.{unknown[0]}",
        )
    path_value = raw.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ConfigError(
            "CFG_MISSING_FIELD",
            "assignment_manifest.path is required",
            "assignment_manifest.path",
        )
    path_value = path_value.strip()
    suffix = Path(path_value).suffix.lower().lstrip(".")
    file_format = str(raw.get("format", suffix)).strip().lower()
    if file_format not in ("csv", "jsonl"):
        raise ConfigError(
            "CFG_ASSIGNMENT_FORMAT",
            f"unsupported assignment_manifest format {file_format}",
            "assignment_manifest.format",
        )
    encoding = str(raw.get("encoding", "utf-8")).strip().lower()
    if encoding not in ("utf-8", "utf-8-sig", "gbk"):
        raise ConfigError(
            "CFG_ENCODING_UNKNOWN",
            f"unsupported encoding {encoding}",
            "assignment_manifest.encoding",
        )
    candidate = Path(path_value)
    resolved = candidate if candidate.is_absolute() else (base / candidate).resolve()
    return AssignmentManifestSpec(resolved, path_value, file_format, encoding)


def _reconcile_manifest(
    rows: list[tuple[str, str, int]],
    findings: list[ValidationFinding],
    config: "Config",
) -> tuple[frozenset[tuple[str, str]], list[ValidationFinding]]:
    pairs: set[tuple[str, str]] = set()
    seen: dict[tuple[str, str], int] = {}
    allowed = set(config.annotators)
    unknown_reported: set[tuple[str, str]] = set()
    for annotator_id, item_id, line_no in rows:
        pair = (annotator_id, item_id)
        if pair in seen:
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_DUPLICATE_PAIR", "error",
                    f"duplicate assignment for {annotator_id}/{item_id}; "
                    f"first at line {seen[pair]}",
                    item_id=item_id, annotator_id=annotator_id, line_no=line_no,
                )
            )
            continue
        seen[pair] = line_no
        pairs.add(pair)
        if allowed and annotator_id not in allowed and pair not in unknown_reported:
            unknown_reported.add(pair)
            findings.append(
                ValidationFinding(
                    "ASSIGNMENT_UNKNOWN_ANNOTATOR", "error",
                    f"manifest annotator {annotator_id} is not in the annotators whitelist",
                    item_id=item_id, annotator_id=annotator_id, line_no=line_no,
                )
            )
    return frozenset(pairs), findings


def build_assignment_plan(
    config: "Config", dataset: Dataset
) -> AssignmentPlan:
    """Resolve the authoritative expected annotator/item pair set."""
    spec = config.assignment_manifest
    if spec is not None:
        rows, digest, findings = load_manifest_pairs(spec)
        pairs, findings = _reconcile_manifest(rows, findings, config)
        return AssignmentPlan(
            "manifest", pairs, tuple(findings), spec.raw_path, digest
        )
    if config.annotators:
        pairs = frozenset(
            (annotator_id, item_id)
            for annotator_id in config.annotators
            for item_id in dataset.item_ids
        )
        return AssignmentPlan("full", pairs, ())
    return AssignmentPlan("discovery", frozenset(), ())


def assignment_coverage(
    dataset: Dataset, plan: AssignmentPlan
) -> dict[str, Any]:
    """Coverage counters. Discovery mode cannot audit absence and reports None."""
    present = {(row.annotator_id, row.item_id) for row in dataset.annotations}
    empty_pairs = {
        (row.annotator_id, row.item_id)
        for row in dataset.annotations if row.is_missing
    }
    expected = set(plan.pairs)
    submitted = present & expected
    explicit_empty = empty_pairs & expected
    completed = submitted - explicit_empty
    missing = expected - present
    unassigned = (
        sum(1 for pair in present if pair not in expected)
        if plan.mode == "manifest"
        else 0
    )
    if plan.mode == "discovery" or not expected:
        coverage_rate = None
        completion_rate = None
    else:
        coverage_rate = len(submitted) / len(expected)
        completion_rate = len(completed) / len(expected)
    return {
        "mode": plan.mode,
        "manifest_path": plan.manifest_path,
        "manifest_sha256": plan.manifest_sha256,
        "expected_pairs": len(expected),
        "submitted_pairs": len(submitted),
        "explicit_empty_pairs": len(explicit_empty),
        "completed_pairs": len(completed),
        "missing_pairs": len(missing),
        "unassigned_records": unassigned,
        "coverage_rate": coverage_rate,
        "completion_rate": completion_rate,
    }


def coverage_findings(
    dataset: Dataset, plan: AssignmentPlan
) -> list[ValidationFinding]:
    """Reconcile actual rows against the authoritative expected pair set."""
    findings: list[ValidationFinding] = []
    present = {(row.annotator_id, row.item_id) for row in dataset.annotations}
    expected = set(plan.pairs)
    for annotator_id, item_id in sorted(expected - present):
        findings.append(
            ValidationFinding(
                "ASSIGNMENT_MISSING",
                "warning",
                f"assigned annotation missing for {annotator_id}/{item_id}",
                item_id=item_id,
                annotator_id=annotator_id,
            )
        )
    if plan.mode == "manifest":
        for row in dataset.annotations:
            if (row.annotator_id, row.item_id) not in expected:
                findings.append(
                    ValidationFinding(
                        "UNASSIGNED_ANNOTATION",
                        "error",
                        f"record for {row.annotator_id}/{row.item_id} is not "
                        "declared in the assignment manifest",
                        item_id=row.item_id,
                        annotator_id=row.annotator_id,
                        line_no=row.line_no,
                    )
                )
    return findings
