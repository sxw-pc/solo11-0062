from __future__ import annotations

from tagconsensus.config import load_config
from tagconsensus.items import (
    annotator_stats,
    build_items,
    leave_one_out,
    majority_reference,
    review_queue,
)
from tagconsensus.metrics import compute_metric

from conftest import make_config, make_dataset


def _five_item_rows():
    rows = []
    values = {
        "it_0001": ["topic_alpha", "topic_alpha", "topic_alpha",
                    "topic_beta", "topic_beta"],
        "it_0002": ["topic_alpha", "topic_alpha", "topic_beta",
                    "topic_beta", "topic_beta"],
        "it_0003": ["topic_alpha", "topic_beta", "other",
                    "topic_alpha", "topic_beta"],
        "it_0004": ["topic_alpha", "topic_beta", "topic_gamma",
                    "topic_delta", "other"],
    }
    for item_id, labels in values.items():
        for index, label in enumerate(labels, start=1):
            rows.append((f"A{index:02d}", item_id, label))
    return rows


def test_threshold_equality_is_accepted(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=3,
        review_threshold=0.6, annotators=[f"A{i:02d}" for i in range(1, 6)],
    ))
    items = {item.item_id: item for item in build_items(make_dataset(_five_item_rows()), config)}
    first = items["it_0001"]
    assert first.vote_rate == 0.6
    assert first.consensus_labels == ("topic_alpha",)
    assert first.needs_review is False


def test_tie_and_low_consensus_are_flagged():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        config = load_config(make_config(
            tmp, metric="pairwise_agreement", min_annotators=3,
            review_threshold=0.8, annotators=[f"A{i:02d}" for i in range(1, 6)],
        ))
        items = {item.item_id: item for item in build_items(make_dataset(_five_item_rows()), config)}
        assert "TIE" in items["it_0003"].review_reasons
        assert "LOW_CONSENSUS" in items["it_0002"].review_reasons
        assert items["it_0003"].consensus_labels == ()


def test_insufficient_annotators_is_reported(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=3,
        annotators=["A01", "A02"],
    ))
    item = build_items(make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ]), config)[0]
    assert item.review_reasons[0] == "INSUFFICIENT_ANNOTATORS"
    assert item.consensus_labels == ("topic_alpha",)


def test_item_with_no_valid_annotation_has_insufficient_reason(tmp_path):
    config = load_config(make_config(tmp_path, metric="pairwise_agreement"))
    item = build_items(make_dataset([
        ("A01", "it_1", ""),
        ("A02", "it_1", ""),
    ]), config)[0]
    assert item.annotator_count == 0
    assert item.vote_rate is None
    assert item.review_reasons == ("INSUFFICIENT_ANNOTATORS",)


def test_multi_label_consensus_and_low_consensus(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", multi_label=True,
        min_annotators=2, review_threshold=0.6,
        annotators=["A01", "A02", "A03"],
    ))
    rows = [
        ("A01", "it_1", ["topic_alpha", "topic_beta"]),
        ("A02", "it_1", ["topic_alpha"]),
        ("A03", "it_1", ["topic_alpha", "topic_beta"]),
        ("A01", "it_2", ["topic_alpha"]),
        ("A02", "it_2", ["topic_beta"]),
        ("A03", "it_2", ["other"]),
    ]
    items = {item.item_id: item for item in build_items(make_dataset(rows), config)}
    assert items["it_1"].consensus_labels == ("topic_alpha", "topic_beta")
    assert items["it_1"].needs_review is False
    assert items["it_2"].consensus_labels == ()
    assert items["it_2"].review_reasons == ("LOW_CONSENSUS",)


def test_annotator_stats_tie_exclusion_and_outlier(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        outlier_threshold=0.65, annotators=["A01", "A02", "A03", "A04"],
    ))
    rows = []
    for item_id, labels in {
        "it_1": ["topic_alpha", "topic_alpha", "topic_alpha", "topic_beta"],
        "it_2": ["topic_beta", "topic_beta", "topic_beta", "topic_alpha"],
        "it_3": ["topic_alpha", "topic_beta", "topic_alpha", "topic_beta"],
    }.items():
        for index, label in enumerate(labels, start=1):
            rows.append((f"A{index:02d}", item_id, label))
    dataset = make_dataset(rows)
    items = build_items(dataset, config)
    stats = {entry.annotator_id: entry for entry in annotator_stats(dataset, config, items)}
    assert stats["A01"].agreement_rate == 1.0
    assert stats["A01"].compared_items == 2
    assert stats["A04"].agreement_rate == 0.0
    assert stats["A04"].compared_items == 2
    assert stats["A04"].is_outlier is True


def test_outlier_threshold_is_strict_lower_bound(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", outlier_threshold=0.5,
        annotators=["A01", "A02", "A03"],
    ))
    rows = [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_alpha"),
        ("A03", "it_2", "topic_beta"),
    ]
    dataset = make_dataset(rows)
    items = build_items(dataset, config)
    stats = {entry.annotator_id: entry for entry in annotator_stats(dataset, config, items)}
    assert stats["A02"].agreement_rate == 0.5
    assert stats["A02"].is_outlier is False


def test_missing_and_out_of_space_counts(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", annotators=["A01", "A02"],
    ))
    rows = [
        ("A01", "it_1", "topic_alpha"),
        ("A01", "it_2", ""),
        ("A02", "it_1", "not_allowed"),
        ("A02", "it_2", "topic_alpha"),
    ]
    dataset = make_dataset(rows)
    items = build_items(dataset, config)
    stats = {entry.annotator_id: entry for entry in annotator_stats(dataset, config, items)}
    assert stats["A01"].missing_count == 1
    assert stats["A02"].out_of_space_count == 1


def test_majority_reference_none_on_tie(tmp_path):
    config = load_config(make_config(tmp_path))
    item = build_items(make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_beta"),
    ]), config)[0]
    assert majority_reference(item, config) is None


def test_leave_one_out_is_sorted_by_absolute_delta(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", annotators=["A01", "A02", "A03", "A04"],
    ))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_alpha"),
        ("A04", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
        ("A04", "it_2", "topic_alpha"),
    ])
    baseline = compute_metric(dataset, config)
    results = leave_one_out(dataset, config, baseline)
    deltas = [abs(item.delta) for item in results if item.delta is not None]
    assert deltas == sorted(deltas, reverse=True)
    assert results[0].annotator_id == "A04"


def test_review_queue_contains_primary_reason(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8,
    ))
    items = build_items(make_dataset(_five_item_rows()), config)
    queue = review_queue(items)
    assert queue
    assert all("reason" in entry for entry in queue)
    assert [entry["item_id"] for entry in queue] == sorted(
        entry["item_id"] for entry in queue
    )
