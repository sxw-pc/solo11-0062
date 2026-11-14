from __future__ import annotations

import pytest

from tagconsensus.adjudicate import apply_adjudications, load_adjudications
from tagconsensus.config import load_config
from tagconsensus.errors import AdjudicationError
from tagconsensus.items import build_items

from conftest import make_config, make_dataset, write_jsonl


def _base(tmp_path, adjudications):
    path = write_jsonl(tmp_path / "adjudications.jsonl", adjudications)
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
    ))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
    ])
    return config, dataset, path


def test_adjudication_priority_over_majority(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "topic_gamma",
        "reviewer": "reviewer_01", "reason": "reviewed",
    }])
    decisions = load_adjudications(path, config, dataset.item_ids)
    items = build_items(dataset, config)
    records = {record.item_id: record for record in apply_adjudications(items, decisions)}
    assert records["it_1"].source == "adjudicated"
    assert records["it_1"].labels == ("topic_gamma",)
    assert records["it_1"].adjudicated_by == "reviewer_01"
    assert records["it_1"].review_reasons == ("LOW_CONSENSUS",)
    assert records["it_2"].source == "majority"
    assert records["it_2"].labels == ("topic_beta",)


def test_unresolved_item_is_explicit(tmp_path):
    config, dataset, path = _base(tmp_path, [])
    items = build_items(dataset, config)
    records = {record.item_id: record
               for record in apply_adjudications(items, [])}
    assert records["it_1"].source == "unresolved"
    assert records["it_1"].labels == ()


def test_no_adjudication_file_returns_empty(tmp_path):
    config, dataset, _ = _base(tmp_path, [])
    assert load_adjudications(None, config, dataset.item_ids) == []


def test_reapplying_is_stable(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "topic_alpha",
        "reviewer": "reviewer_01", "reason": "reviewed",
    }])
    items = build_items(dataset, config)
    first = apply_adjudications(items, load_adjudications(path, config, dataset.item_ids))
    second = apply_adjudications(items, load_adjudications(path, config, dataset.item_ids))
    assert first == second


def test_duplicate_adjudication_errors(tmp_path):
    config, dataset, path = _base(tmp_path, [
        {"item_id": "it_1", "decision_label": "topic_alpha",
         "reviewer": "reviewer_01", "reason": "one"},
        {"item_id": "it_1", "decision_label": "topic_beta",
         "reviewer": "reviewer_02", "reason": "two"},
    ])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_DUPLICATE_ITEM"


def test_label_out_of_space_errors(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "not_allowed",
        "reviewer": "reviewer_01", "reason": "reviewed",
    }])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_LABEL_OUT_OF_SPACE"


def test_unknown_item_errors(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_404", "decision_label": "topic_alpha",
        "reviewer": "reviewer_01", "reason": "reviewed",
    }])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_UNKNOWN_ITEM"


def test_missing_field_and_malformed_rows(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "topic_alpha",
        "reviewer": "reviewer_01",
    }])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_MISSING_FIELD"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_MALFORMED_ROW"


def test_missing_file_errors(tmp_path):
    config, dataset, _ = _base(tmp_path, [])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(tmp_path / "missing.jsonl", config, dataset.item_ids)
    assert error.value.code == "ADJ_FILE_NOT_FOUND"
