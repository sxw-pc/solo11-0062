"""Per-item consensus, reviewer queue, annotator drift and leave-one-out influence."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from .config import Config
from .errors import MetricError
from .metrics import compute_metric
from .models import (
    AnnotatorStat,
    Dataset,
    ItemConsensus,
    LeaveOneOut,
    MetricResult,
    sorted_counts,
)


def build_items(dataset: Dataset, config: Config) -> list[ItemConsensus]:
    buckets: dict[str, Counter[str]] = {item_id: Counter() for item_id in dataset.item_ids}
    counts: Counter[str] = Counter()
    for row in dataset.annotations:
        counts[row.item_id] += 0 if row.is_missing else 1
        for label in row.labels:
            buckets[row.item_id][label] += 1

    results: list[ItemConsensus] = []
    for item_id in sorted(buckets):
        votes = buckets[item_id]
        annotator_count = counts[item_id]
        ordered = sorted_counts(votes)
        if annotator_count == 0:
            results.append(
                ItemConsensus(item_id, 0, ordered, (), None,
                              ("INSUFFICIENT_ANNOTATORS",))
            )
            continue
        top_count = max(votes.values()) if votes else 0
        top_labels = sorted(label for label, count in votes.items() if count == top_count)
        reasons: list[str] = []
        if annotator_count < config.min_annotators_per_item:
            reasons.append("INSUFFICIENT_ANNOTATORS")
        if config.multi_label:
            selected = tuple(
                sorted(
                    label
                    for label, count in votes.items()
                    if count / annotator_count >= config.review_threshold
                )
            )
            consensus_labels = selected
            if not selected:
                reasons.append("LOW_CONSENSUS")
            vote_rate = top_count / annotator_count if top_count else 0.0
        else:
            if len(top_labels) > 1:
                reasons.append("TIE")
                consensus_labels = ()
            elif top_labels:
                consensus_labels = (top_labels[0],)
            else:
                consensus_labels = ()
            vote_rate = top_count / annotator_count
            if len(top_labels) == 1 and vote_rate < config.review_threshold:
                reasons.append("LOW_CONSENSUS")
        ordered_reasons = tuple(
            code for code in ("INSUFFICIENT_ANNOTATORS", "TIE", "LOW_CONSENSUS")
            if code in reasons
        )
        results.append(
            ItemConsensus(
                item_id,
                annotator_count,
                ordered,
                consensus_labels,
                vote_rate,
                ordered_reasons,
            )
        )
    return results


def majority_reference(item: ItemConsensus, config: Config) -> tuple[str, ...] | None:
    if not item.vote_counts:
        return None
    top_count = max(count for _, count in item.vote_counts)
    if top_count <= 0:
        return None
    top_labels = tuple(label for label, count in item.vote_counts if count == top_count)
    if not config.multi_label and len(top_labels) > 1:
        return None
    return top_labels


def annotator_stats(
    dataset: Dataset, config: Config, items: Iterable[ItemConsensus]
) -> list[AnnotatorStat]:
    item_map = {item.item_id: item for item in items}
    rows: dict[str, list] = {annotator: [] for annotator in dataset.annotators}
    for row in dataset.annotations:
        rows[row.annotator_id].append(row)
    allowed = set(config.label_space)
    results: list[AnnotatorStat] = []
    for annotator_id in sorted(rows):
        annotator_rows = rows[annotator_id]
        participated = {row.item_id for row in annotator_rows}
        missing_count = sum(1 for row in annotator_rows if row.is_missing)
        out_of_space = sum(
            1 for row in annotator_rows for label in row.labels if label not in allowed
        )
        compared = 0
        agreed = 0
        for row in annotator_rows:
            if row.is_missing:
                continue
            reference = majority_reference(item_map[row.item_id], config)
            if reference is None:
                continue
            compared += 1
            if tuple(sorted(set(row.labels))) == reference:
                agreed += 1
        rate = agreed / compared if compared else None
        outlier = rate is not None and rate < config.outlier_threshold
        results.append(
            AnnotatorStat(
                annotator_id,
                rate,
                compared,
                len(participated),
                missing_count,
                out_of_space,
                outlier,
            )
        )
    return results


def review_queue(items: Iterable[ItemConsensus]) -> list[dict]:
    records = []
    for item in items:
        if not item.needs_review:
            continue
        record = item.to_dict()
        record["reason"] = item.primary_reason
        records.append(record)
    return sorted(records, key=lambda record: record["item_id"])


def leave_one_out(
    dataset: Dataset, config: Config, baseline: MetricResult
) -> list[LeaveOneOut]:
    results: list[LeaveOneOut] = []
    for annotator_id in dataset.annotators:
        try:
            result = compute_metric(
                dataset.without_annotator(annotator_id), config, bootstrap=False
            )
            delta = result.value - baseline.value
            results.append(LeaveOneOut(annotator_id, result.value, delta, None))
        except MetricError as exc:
            results.append(LeaveOneOut(annotator_id, None, None, exc.code))
    results.sort(
        key=lambda entry: (
            entry.delta is None,
            -abs(entry.delta) if entry.delta is not None else 0.0,
            entry.annotator_id,
        )
    )
    return results
