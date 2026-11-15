"""Immutable data models shared by loading, statistics, reports and the ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .hashing import annotator_fingerprint, sha256_text, stable_json

REVIEW_REASONS = ("INSUFFICIENT_ANNOTATORS", "TIE", "LOW_CONSENSUS")


@dataclass(frozen=True)
class ValidationFinding:
    code: str
    severity: str
    message: str
    item_id: str | None = None
    annotator_id: str | None = None
    line_no: int | None = None
    promoted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "item_id": self.item_id,
            "annotator_id": self.annotator_id,
            "line_no": self.line_no,
            "promoted": self.promoted,
        }


@dataclass(frozen=True)
class Annotation:
    annotator_id: str
    item_id: str
    labels: tuple[str, ...]
    confidence: float | None
    line_no: int

    @property
    def is_missing(self) -> bool:
        return not self.labels


@dataclass(frozen=True)
class Dataset:
    annotations: tuple[Annotation, ...]
    source_path: str
    file_format: str
    file_sha256: str
    parse_findings: tuple[ValidationFinding, ...] = field(default_factory=tuple)

    @property
    def records(self) -> int:
        return len(self.annotations)

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.item_id for row in self.annotations}))

    @property
    def annotators(self) -> tuple[str, ...]:
        return tuple(sorted({row.annotator_id for row in self.annotations}))

    @property
    def annotator_fingerprint(self) -> str:
        return annotator_fingerprint(self.annotators)

    @property
    def missing_annotations(self) -> int:
        return sum(1 for row in self.annotations if row.is_missing)

    def without_annotator(self, annotator_id: str) -> "Dataset":
        kept = tuple(
            row for row in self.annotations if row.annotator_id != annotator_id
        )
        return Dataset(
            annotations=kept,
            source_path=self.source_path,
            file_format=self.file_format,
            file_sha256=self.file_sha256,
            parse_findings=self.parse_findings,
        )

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "record_count": self.records,
            "annotator_fingerprint": self.annotator_fingerprint,
        }

    def fingerprint(self) -> str:
        return sha256_text(stable_json(self.fingerprint_dict()))


RESAMPLE_ID_MARKER = "#bootstrap#"


def resample_dataset(dataset: Dataset, item_ids: Iterable[str]) -> Dataset:
    """Build an item-cluster bootstrap resample as a new immutable Dataset.

    ``item_ids`` is one bootstrap draw: as many entries as the original item
    count, sampled with replacement, so the same original item may appear
    several times. Each entry becomes an *independent occurrence* carrying all
    annotations of its source cluster. Because ``Dataset.item_ids`` deduplicates
    by item_id and the metric layer buckets annotations by item_id, every
    occurrence receives a synthetic item_id (``<item_id>#bootstrap#<k>`` with
    ``k`` the draw position, unique within the draw); duplicate draws of one
    cluster are therefore never merged back into a single unit.

    The source dataset is not modified; file metadata is carried over so the
    resample stays traceable to its origin.
    """
    by_item: dict[str, list[Annotation]] = {}
    for row in dataset.annotations:
        by_item.setdefault(row.item_id, []).append(row)
    annotations: list[Annotation] = []
    for occurrence, item_id in enumerate(item_ids):
        synthetic_id = f"{item_id}{RESAMPLE_ID_MARKER}{occurrence}"
        for row in by_item.get(item_id, ()):
            annotations.append(
                Annotation(
                    row.annotator_id,
                    synthetic_id,
                    row.labels,
                    row.confidence,
                    row.line_no,
                )
            )
    return Dataset(
        annotations=tuple(annotations),
        source_path=dataset.source_path,
        file_format=dataset.file_format,
        file_sha256=dataset.file_sha256,
        parse_findings=dataset.parse_findings,
    )


@dataclass(frozen=True)
class BootstrapInterval:
    """Fixed-seed item-cluster percentile bootstrap result for one metric run.

    ``requested`` is the configured replicate count and never changes, even
    when some draws are degenerate; degenerate draws are counted in
    ``degenerate`` (broken down by error code in ``degenerate_codes``) and
    excluded from the percentile distribution instead of being treated as 0
    or silently re-drawn. ``status`` is "ok" when the valid ratio meets
    ``min_valid_ratio``; otherwise "unavailable" and the bounds are None with
    ``unavailable_reason`` explaining why.
    """

    seed: int
    confidence: float
    requested: int
    valid: int
    degenerate: int
    min_valid_ratio: float
    status: str
    lower: float | None
    upper: float | None
    unavailable_reason: str | None
    degenerate_codes: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "item_cluster_percentile",
            "seed": self.seed,
            "confidence": self.confidence,
            "requested": self.requested,
            "valid": self.valid,
            "degenerate": self.degenerate,
            "min_valid_ratio": self.min_valid_ratio,
            "status": self.status,
            "lower": self.lower,
            "upper": self.upper,
            "unavailable_reason": self.unavailable_reason,
            "degenerate_codes": {code: count for code, count in self.degenerate_codes},
        }


@dataclass(frozen=True)
class LabelMetric:
    label: str | None
    value: float | None
    po: float | None = None
    pe: float | None = None
    do: float | None = None
    de: float | None = None
    eligible_items: int = 0
    eligible_annotations: int = 0
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value": self.value,
            "po": self.po,
            "pe": self.pe,
            "Do": self.do,
            "De": self.de,
            "eligible_items": self.eligible_items,
            "eligible_annotations": self.eligible_annotations,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    po: float | None
    pe: float | None
    do: float | None
    de: float | None
    eligible_items: int
    eligible_annotations: int
    per_label: tuple[LabelMetric, ...] = field(default_factory=tuple)
    macro_average: float | None = None
    bootstrap: BootstrapInterval | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "po": self.po,
            "pe": self.pe,
            "Do": self.do,
            "De": self.de,
            "eligible_items": self.eligible_items,
            "eligible_annotations": self.eligible_annotations,
            "macro_average": self.macro_average,
            "per_label": [entry.to_dict() for entry in self.per_label],
            "bootstrap": self.bootstrap.to_dict() if self.bootstrap else None,
        }


@dataclass(frozen=True)
class ItemConsensus:
    item_id: str
    annotator_count: int
    vote_counts: tuple[tuple[str, int], ...]
    consensus_labels: tuple[str, ...]
    vote_rate: float | None
    review_reasons: tuple[str, ...] = ()

    @property
    def primary_reason(self) -> str | None:
        return self.review_reasons[0] if self.review_reasons else None

    @property
    def needs_review(self) -> bool:
        return bool(self.review_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "annotator_count": self.annotator_count,
            "vote_counts": {key: value for key, value in self.vote_counts},
            "consensus_labels": list(self.consensus_labels),
            "vote_rate": self.vote_rate,
            "review_reasons": list(self.review_reasons),
        }


@dataclass(frozen=True)
class AnnotatorStat:
    annotator_id: str
    agreement_rate: float | None
    compared_items: int
    participation_items: int
    missing_count: int
    out_of_space_count: int
    is_outlier: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "agreement_rate": self.agreement_rate,
            "compared_items": self.compared_items,
            "participation_items": self.participation_items,
            "missing_count": self.missing_count,
            "out_of_space_count": self.out_of_space_count,
            "is_outlier": self.is_outlier,
        }


@dataclass(frozen=True)
class Adjudication:
    item_id: str
    decision_label: str
    reviewer: str
    reason: str
    line_no: int


@dataclass(frozen=True)
class ConsensusRecord:
    item_id: str
    labels: tuple[str, ...]
    source: str
    vote_rate: float | None
    review_reasons: tuple[str, ...]
    adjudicated_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "labels": list(self.labels),
            "source": self.source,
            "vote_rate": self.vote_rate,
            "review_reasons": list(self.review_reasons),
            "adjudicated_by": self.adjudicated_by,
        }


@dataclass(frozen=True)
class LeaveOneOut:
    annotator_id: str
    value: float | None
    delta: float | None
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotator_id": self.annotator_id,
            "value": self.value,
            "delta": self.delta,
            "absolute_delta": None if self.delta is None else abs(self.delta),
            "error_code": self.error_code,
        }


def sorted_counts(counts: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items(), key=lambda pair: pair[0]))


def as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))
