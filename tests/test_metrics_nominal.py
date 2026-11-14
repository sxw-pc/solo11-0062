from __future__ import annotations

import json

import pytest

from tagconsensus.config import load_config
from tagconsensus.errors import MetricError
from tagconsensus.metrics import compute_metric

from conftest import PROJECT_ROOT, make_config, make_dataset

GOLDEN = json.loads(
    (PROJECT_ROOT / "data" / "expected" / "demo_golden.json").read_text(encoding="utf-8")
)


def _cohen_dataset(tmp_path):
    sample = GOLDEN["cohen_2x2"]
    rows = []
    for index, (left, right) in enumerate(
        zip(sample["annotator_a"], sample["annotator_b"])
    ):
        rows.append(("A01", f"it_{index:04d}", left))
        rows.append(("A02", f"it_{index:04d}", right))
    return make_dataset(rows)


def _rating_dataset(tmp_path):
    rows = []
    for item_index, values in enumerate(GOLDEN["rating_3x4"]["items"]):
        for annotator_index, value in enumerate(values):
            rows.append((f"A{annotator_index + 1:02d}", f"it_{item_index:04d}", value))
    return make_dataset(rows)


def test_cohen_classic_2x2_exact_values(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02"]
    ))
    result = compute_metric(_cohen_dataset(tmp_path), config)
    expected = GOLDEN["cohen_2x2"]["expected"]
    assert result.value == pytest.approx(expected["value"], abs=1e-9)
    assert result.po == pytest.approx(expected["po"], abs=1e-9)
    assert result.pe == pytest.approx(expected["pe"], abs=1e-9)
    assert result.do == pytest.approx(expected["Do"], abs=1e-9)
    assert result.de == pytest.approx(expected["De"], abs=1e-9)
    assert result.eligible_items == 10
    assert result.eligible_annotations == 20


def test_fleiss_classic_3x4_exact_value(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="fleiss", annotators=["A01", "A02", "A03"]
    ))
    result = compute_metric(_rating_dataset(tmp_path), config)
    expected = GOLDEN["rating_3x4"]["expected"]["fleiss"]
    assert result.value == pytest.approx(expected, abs=1e-9)
    assert result.po == pytest.approx(2 / 3, abs=1e-9)
    assert result.pe == pytest.approx(0.5, abs=1e-9)
    assert result.do == pytest.approx(1 / 3, abs=1e-9)
    assert result.de == pytest.approx(0.5, abs=1e-9)


def test_krippendorff_classic_3x4_exact_value(tmp_path):
    config = load_config(make_config(
        tmp_path, metric="krippendorff_nominal", annotators=["A01", "A02", "A03"]
    ))
    result = compute_metric(_rating_dataset(tmp_path), config)
    expected = GOLDEN["rating_3x4"]["expected"]["krippendorff_nominal"]
    assert result.value == pytest.approx(expected, abs=1e-9)
    assert result.do == pytest.approx(1 / 3, abs=1e-9)
    assert result.de == pytest.approx(6 / 11, abs=1e-9)
    assert result.pe == pytest.approx(5 / 11, abs=1e-9)


def test_pairwise_classic_3x4_exact_value(tmp_path):
    config = load_config(make_config(tmp_path, metric="pairwise_agreement"))
    result = compute_metric(_rating_dataset(tmp_path), config)
    expected = GOLDEN["rating_3x4"]["expected"]["pairwise_agreement"]
    assert result.value == pytest.approx(expected, abs=1e-9)
    assert result.po == pytest.approx(expected, abs=1e-9)
    assert result.pe is None
    assert result.do == pytest.approx(1 / 3, abs=1e-9)
    assert result.eligible_items == 4
    assert result.eligible_annotations == 12


def test_krippendorff_tolerates_missing_annotations(tmp_path):
    rows = []
    values = [["topic_alpha", "topic_alpha", "topic_beta"],
              ["topic_alpha", None, "topic_alpha"],
              ["topic_beta", "topic_beta", "topic_beta"]]
    for item_index, item_values in enumerate(values):
        for annotator_index, value in enumerate(item_values):
            if value is not None:
                rows.append((f"A{annotator_index + 1:02d}",
                             f"it_{item_index}", value))
    config = load_config(make_config(
        tmp_path, metric="krippendorff_nominal",
        annotators=["A01", "A02", "A03"],
    ))
    result = compute_metric(make_dataset(rows), config)
    assert -1.0 <= result.value <= 1.0
    assert result.eligible_items == 3
    assert result.eligible_annotations == 8


def test_cohen_pair_selection_with_more_than_two_annotators(tmp_path):
    rows = []
    for index in range(4):
        shared = "topic_alpha" if index < 2 else "topic_beta"
        rows.extend([
            ("A01", f"it_{index}", shared),
            ("A02", f"it_{index}", shared),
            ("A03", f"it_{index}", "topic_gamma"),
        ])
    config = load_config(make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02", "A03"],
        extra={"pair": ["A01", "A02"]},
    ))
    result = compute_metric(make_dataset(rows), config)
    assert result.value == pytest.approx(1.0, abs=1e-9)
    assert result.eligible_annotations == 8


def test_multi_label_reports_per_label_and_macro(tmp_path):
    rows = [
        ("A01", "it_1", ["topic_alpha", "topic_beta"]),
        ("A02", "it_1", ["topic_alpha"]),
        ("A03", "it_1", ["topic_alpha", "topic_beta"]),
        ("A01", "it_2", ["topic_beta"]),
        ("A02", "it_2", ["topic_beta"]),
        ("A03", "it_2", ["topic_alpha"]),
        ("A01", "it_3", ["topic_beta"]),
        ("A02", "it_3", []),
        ("A03", "it_3", ["topic_beta"]),
    ]
    config = load_config(make_config(
        tmp_path, metric="krippendorff_nominal", multi_label=True,
        annotators=["A01", "A02", "A03"],
    ))
    result = compute_metric(make_dataset(rows), config)
    assert len(result.per_label) == 5
    labels = {entry.label: entry for entry in result.per_label}
    assert labels["topic_alpha"].value is not None
    assert labels["topic_beta"].value is not None
    assert labels["topic_gamma"].error_code == "METRIC_PE_ONE"
    valid = [entry.value for entry in result.per_label if entry.value is not None]
    assert result.macro_average == pytest.approx(sum(valid) / len(valid), abs=1e-12)
    assert result.value == result.macro_average


def test_pairwise_without_two_annotations_per_item_is_no_items(tmp_path):
    config = load_config(make_config(tmp_path, metric="pairwise_agreement"))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    with pytest.raises(MetricError) as error:
        compute_metric(dataset, config)
    assert error.value.code == "METRIC_NO_ITEMS"
