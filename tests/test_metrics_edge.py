from __future__ import annotations

import pytest

from tagconsensus.config import load_config
from tagconsensus.errors import MetricError
from tagconsensus.metrics import compute_metric

from conftest import make_config, make_dataset


def _perfect_rows(annotators=("A01", "A02", "A03")):
    rows = []
    for item_index, label in enumerate(
        ["topic_alpha", "topic_beta", "topic_alpha", "topic_beta"]
    ):
        for annotator in annotators:
            rows.append((annotator, f"it_{item_index}", label))
    return rows


def test_perfect_agreement_is_one_for_supported_metrics(tmp_path):
    for metric in ("fleiss", "krippendorff_nominal", "pairwise_agreement"):
        config = load_config(make_config(
            tmp_path, metric=metric, annotators=["A01", "A02", "A03"]
        ))
        result = compute_metric(make_dataset(_perfect_rows()), config)
        assert result.value == pytest.approx(1.0, abs=1e-9), metric


def test_perfect_agreement_for_cohen_is_one(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02"]
    ))
    result = compute_metric(make_dataset(_perfect_rows(("A01", "A02"))), config)
    assert result.value == pytest.approx(1.0, abs=1e-9)


def test_cohen_chance_agreement_is_zero(tmp_path):
    rows = []
    for index, (left, right) in enumerate([
        ("topic_alpha", "topic_alpha"),
        ("topic_alpha", "topic_beta"),
        ("topic_beta", "topic_alpha"),
        ("topic_beta", "topic_beta"),
    ]):
        rows.append(("A01", f"it_{index}", left))
        rows.append(("A02", f"it_{index}", right))
    config = load_config(make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02"]
    ))
    result = compute_metric(make_dataset(rows), config)
    assert result.po == pytest.approx(0.5, abs=1e-9)
    assert result.pe == pytest.approx(0.5, abs=1e-9)
    assert abs(result.value) <= 1e-9


def test_single_effective_item_pairwise(tmp_path):
    config = load_config(make_config(tmp_path, metric="pairwise_agreement"))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ])
    result = compute_metric(dataset, config)
    assert result.value == 1.0
    assert result.eligible_items == 1
    assert result.eligible_annotations == 2


@pytest.mark.parametrize("metric", [
    "cohen", "fleiss", "krippendorff_nominal", "pairwise_agreement"
])
def test_all_missing_has_no_eligible_items(tmp_path, metric):
    config = load_config(make_config(
        tmp_path, metric=metric, annotators=["A01", "A02"]
    ))
    dataset = make_dataset([
        ("A01", "it_1", ""),
        ("A02", "it_1", ""),
    ])
    with pytest.raises(MetricError) as error:
        compute_metric(dataset, config)
    assert error.value.code == "METRIC_NO_ITEMS"


def test_fleiss_unbalanced_raises_without_number(tmp_path):
    rows = []
    for annotator in ("A01", "A02", "A03"):
        rows.append((annotator, "it_1", "topic_alpha"))
    for annotator in ("A01", "A02"):
        rows.append((annotator, "it_2", "topic_beta"))
    config = load_config(make_config(
        tmp_path, metric="fleiss", annotators=["A01", "A02", "A03"]
    ))
    with pytest.raises(MetricError) as error:
        compute_metric(make_dataset(rows), config)
    assert error.value.code == "METRIC_UNBALANCED"


@pytest.mark.parametrize("metric", ["cohen", "fleiss", "krippendorff_nominal"])
def test_pe_one_is_an_explicit_error(tmp_path, metric):
    rows = []
    for index in range(3):
        rows.append(("A01", f"it_{index}", "topic_alpha"))
        rows.append(("A02", f"it_{index}", "topic_alpha"))
    config = load_config(make_config(
        tmp_path, metric=metric, annotators=["A01", "A02"]
    ))
    with pytest.raises(MetricError) as error:
        compute_metric(make_dataset(rows), config)
    assert error.value.code == "METRIC_PE_ONE"


def test_pairwise_single_label_universe_is_one(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", annotators=["A01", "A02"]
    ))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ])
    assert compute_metric(dataset, config).value == 1.0


def test_cohen_more_than_two_annotators_without_pair(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02", "A03"]
    ))
    with pytest.raises(MetricError) as error:
        compute_metric(make_dataset(_perfect_rows()), config)
    assert error.value.code == "METRIC_ANNOTATOR_COUNT"


def test_empty_dataset_values_are_errors(tmp_path):
    config = load_config(make_config(tmp_path, metric="krippendorff_nominal"))
    with pytest.raises(MetricError) as error:
        compute_metric(make_dataset([]), config)
    assert error.value.code == "METRIC_NO_ITEMS"


def test_multi_label_all_degenerate_raises(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", multi_label=True,
        annotators=["A01", "A02"],
    ))
    rows = []
    for index in range(2):
        rows.append(("A01", f"it_{index}", "topic_alpha"))
        rows.append(("A02", f"it_{index}", "topic_alpha"))
    result = compute_metric(make_dataset(rows), config)
    assert result.value == 1.0
    assert all(entry.value is not None for entry in result.per_label)
