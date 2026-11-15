"""Self-contained nominal agreement statistics with explicit degeneracy handling."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable, Sequence

from .config import Config
from .errors import MetricError
from .models import LabelMetric, MetricResult, RecordSource

EPSILON = 1e-12


def _items(source: RecordSource) -> list[str]:
    return list(source.item_ids)


def _single_units(source: RecordSource) -> list[Counter[str]]:
    buckets: dict[str, Counter[str]] = {item_id: Counter() for item_id in _items(source)}
    for row in source.iter_annotations():
        if not row.is_missing:
            buckets[row.item_id][row.labels[0]] += 1
    return [buckets[item_id] for item_id in sorted(buckets)]


def _binary_units(source: RecordSource, label: str) -> list[Counter[int]]:
    buckets: dict[str, Counter[int]] = {item_id: Counter() for item_id in _items(source)}
    for row in source.iter_annotations():
        if row.is_missing:
            continue
        value = 1 if label in row.labels else 0
        buckets[row.item_id][value] += 1
    return [buckets[item_id] for item_id in sorted(buckets)]


def _binary_value(row_labels: Sequence[str], label: str) -> int:
    return 1 if label in row_labels else 0


def _pairwise(units: list[Counter]) -> LabelMetric:
    total_pairs = 0.0
    agreeing_pairs = 0.0
    annotations = 0
    items = 0
    for counts in units:
        size = sum(counts.values())
        if size < 2:
            continue
        items += 1
        annotations += size
        total_pairs += size * (size - 1) / 2.0
        for count in counts.values():
            agreeing_pairs += count * (count - 1) / 2.0
    if total_pairs == 0.0:
        raise MetricError("METRIC_NO_ITEMS", "no item has at least two annotations")
    po = agreeing_pairs / total_pairs
    return LabelMetric(
        None, po, po=po, do=1.0 - po, eligible_items=items,
        eligible_annotations=annotations
    )


def _check_pe(pe: float, do: float) -> float:
    if abs(1.0 - pe) <= EPSILON:
        raise MetricError("METRIC_PE_ONE", "expected agreement is 1.0; kappa is undefined")
    return 1.0 - do / (1.0 - pe)


def _cohen(pairs: list[tuple[Hashable, Hashable]]) -> LabelMetric:
    total = len(pairs)
    agreeing = 0
    left_counts: Counter = Counter()
    right_counts: Counter = Counter()
    for left, right in pairs:
        left_counts[left] += 1
        right_counts[right] += 1
        if left == right:
            agreeing += 1
    if total == 0:
        raise MetricError("METRIC_NO_ITEMS", "no complete pair of annotations")
    po = agreeing / total
    pe = sum(
        (left_counts[key] / total) * (right_counts[key] / total)
        for key in set(left_counts) | set(right_counts)
    )
    value = _check_pe(pe, 1.0 - po)
    return LabelMetric(
        None, value, po=po, pe=pe, do=1.0 - po, de=1.0 - pe,
        eligible_items=total, eligible_annotations=total * 2
    )


def _fleiss(units: list[Counter]) -> LabelMetric:
    sizes = [sum(counts.values()) for counts in units]
    if not sizes or max(sizes) == 0:
        raise MetricError("METRIC_NO_ITEMS", "dataset contains no ratings")
    per_item = sizes[0]
    if per_item < 2 or any(size != per_item for size in sizes):
        raise MetricError(
            "METRIC_UNBALANCED", "Fleiss requires the same ratings count for every item"
        )
    item_agreements = []
    totals: Counter = Counter()
    for counts in units:
        item_agreements.append(
            (sum(count * count for count in counts.values()) - per_item)
            / (per_item * (per_item - 1))
        )
        totals.update(counts)
    po = sum(item_agreements) / len(units)
    assignments = len(units) * per_item
    pe = sum((totals[key] / assignments) ** 2 for key in totals)
    value = _check_pe(pe, 1.0 - po)
    return LabelMetric(
        None, value, po=po, pe=pe, do=1.0 - po, de=1.0 - pe,
        eligible_items=len(units), eligible_annotations=assignments
    )


def _krippendorff(units: list[Counter]) -> LabelMetric:
    eligible = [counts for counts in units if sum(counts.values()) >= 2]
    if not eligible:
        raise MetricError("METRIC_NO_ITEMS", "no item has at least two annotations")
    total_n = sum(sum(counts.values()) for counts in eligible)
    diagonal = 0.0
    marginals: Counter = Counter()
    for counts in eligible:
        size = sum(counts.values())
        marginals.update(counts)
        for count in counts.values():
            diagonal += count * (count - 1) / (size - 1)
    do = 1.0 - diagonal / total_n
    de = (
        total_n * total_n - sum(count * count for count in marginals.values())
    ) / (total_n * (total_n - 1))
    if de <= EPSILON:
        raise MetricError("METRIC_PE_ONE", "expected disagreement is zero; alpha is undefined")
    alpha = 1.0 - do / de
    return LabelMetric(
        None, alpha, po=1.0 - do, pe=1.0 - de, do=do, de=de,
        eligible_items=len(eligible), eligible_annotations=total_n
    )


def _cohen_pairs(
    source: RecordSource, config: Config, label: str | None
) -> list[tuple[Hashable, Hashable]]:
    annotators = set(source.annotators)
    if config.pair is not None:
        pair = config.pair
        if any(name not in annotators for name in pair):
            raise MetricError(
                "METRIC_ANNOTATOR_COUNT", "pair contains an annotator absent from the data"
            )
    else:
        if len(annotators) != 2:
            raise MetricError(
                "METRIC_ANNOTATOR_COUNT", "cohen requires exactly two annotators or a pair"
            )
        pair = (source.annotators[0], source.annotators[1])
    maps: dict[str, dict[str, Hashable]] = {pair[0]: {}, pair[1]: {}}
    for row in source.iter_annotations():
        if row.annotator_id not in maps or row.is_missing:
            continue
        value = row.labels[0] if label is None else _binary_value(row.labels, label)
        maps[row.annotator_id][row.item_id] = value
    return [
        (maps[pair[0]][item_id], maps[pair[1]][item_id])
        for item_id in sorted(set(maps[pair[0]]) & set(maps[pair[1]]))
    ]


def _compute_one(source: RecordSource, config: Config, label: str | None) -> LabelMetric:
    if config.metric == "cohen":
        result = _cohen(_cohen_pairs(source, config, label))
    elif config.metric == "fleiss":
        units = _single_units(source) if label is None else _binary_units(source, label)
        result = _fleiss(units)
    elif config.metric == "krippendorff_nominal":
        units = _single_units(source) if label is None else _binary_units(source, label)
        result = _krippendorff(units)
    elif config.metric == "pairwise_agreement":
        units = _single_units(source) if label is None else _binary_units(source, label)
        result = _pairwise(units)
    else:
        raise MetricError("METRIC_UNKNOWN", f"unsupported metric {config.metric}")
    return LabelMetric(
        label, result.value, result.po, result.pe, result.do, result.de,
        result.eligible_items, result.eligible_annotations, result.error_code
    )


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def compute_metric(source: RecordSource, config: Config) -> MetricResult:
    if not config.multi_label:
        result = _compute_one(source, config, None)
        return MetricResult(
            config.metric, result.value, result.po, result.pe, result.do, result.de,
            result.eligible_items, result.eligible_annotations
        )

    per_label: list[LabelMetric] = []
    for label in config.label_space:
        try:
            per_label.append(_compute_one(source, config, label))
        except MetricError as exc:
            per_label.append(LabelMetric(label, None, error_code=exc.code))
    valid = [entry for entry in per_label if entry.value is not None]
    if not valid:
        raise MetricError("METRIC_ALL_LABELS_INVALID", "no label produced a usable metric")
    macro = sum(entry.value for entry in valid if entry.value is not None) / len(valid)
    eligible_items = len(source.item_ids)
    eligible_annotations = sum(
        1 for row in source.iter_annotations() if not row.is_missing
    )
    return MetricResult(
        config.metric,
        macro,
        _mean(entry.po for entry in valid),
        _mean(entry.pe for entry in valid),
        _mean(entry.do for entry in valid),
        _mean(entry.de for entry in valid),
        eligible_items,
        eligible_annotations,
        tuple(per_label),
        macro,
    )
