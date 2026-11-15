from __future__ import annotations

import json

import pytest

from tagconsensus.adjudicate import (
    apply_adjudications,
    build_evidence_basis,
    load_adjudications,
    verify_adjudications,
)
from tagconsensus.config import load_config
from tagconsensus.errors import AdjudicationError
from tagconsensus.items import build_items

from conftest import bind_decisions, make_config, make_dataset, write_jsonl

LABELS = ["topic_alpha", "topic_beta", "topic_gamma", "topic_delta", "other"]


def _base(tmp_path, adjudications, **config_overrides):
    path = write_jsonl(tmp_path / "adjudications.jsonl", adjudications)
    config = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"], **config_overrides,
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


def _bound_base(tmp_path, decisions=("it_1", "topic_gamma", "reviewer_01", "reviewed"),
                **config_overrides):
    overrides = {
        "metric": "pairwise_agreement", "min_annotators": 2,
        "review_threshold": 0.8, "annotators": ["A01", "A02", "A03"],
    }
    overrides.update(config_overrides)
    config = load_config(make_config(tmp_path, **overrides))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
    ])
    rows = bind_decisions([decisions], config, dataset)
    path = write_jsonl(tmp_path / "adjudications.jsonl", rows)
    config = load_config(make_config(
        tmp_path, adjudication_file=path, **overrides,
    ))
    return config, dataset, path


def _verify(config, dataset, path):
    basis = build_evidence_basis(dataset, config)
    loaded = load_adjudications(path, config, dataset.item_ids)
    return verify_adjudications(loaded, basis, config)


def test_adjudication_priority_over_majority(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    decisions = _verify(config, dataset, path)
    assert decisions[0].is_valid
    items = build_items(dataset, config)
    records = {record.item_id: record for record in apply_adjudications(items, decisions)}
    assert records["it_1"].source == "adjudicated"
    assert records["it_1"].labels == ("topic_gamma",)
    assert records["it_1"].adjudicated_by == "reviewer_01"
    assert records["it_1"].review_reasons == ("LOW_CONSENSUS",)
    assert records["it_2"].source == "majority"
    assert records["it_2"].labels == ("topic_beta",)


def test_unresolved_item_is_explicit(tmp_path):
    config, dataset, _ = _bound_base(tmp_path)
    items = build_items(dataset, config)
    records = {item.item_id: item
               for item in apply_adjudications(items, [])}
    assert records["it_1"].source == "unresolved"
    assert records["it_1"].labels == ()


def test_no_adjudication_file_returns_empty(tmp_path):
    config, dataset, _ = _bound_base(tmp_path)
    assert load_adjudications(None, config, dataset.item_ids) == []


def test_reapplying_is_stable(tmp_path):
    config, dataset, path = _bound_base(
        tmp_path, ("it_1", "topic_alpha", "reviewer_01", "reviewed")
    )
    items = build_items(dataset, config)
    first = apply_adjudications(items, _verify(config, dataset, path))
    second = apply_adjudications(items, _verify(config, dataset, path))
    assert first == second


def test_duplicate_adjudication_errors(tmp_path):
    config, dataset, _ = _bound_base(tmp_path)
    rows = bind_decisions([
        ("it_1", "topic_alpha", "reviewer_01", "one"),
        ("it_1", "topic_beta", "reviewer_02", "two"),
    ], config, dataset)
    path = write_jsonl(tmp_path / "dup.jsonl", rows)
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_DUPLICATE_ITEM"


def test_label_out_of_space_errors_without_binding(tmp_path):
    # Unbound rows keep the original hard-error behavior.
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "not_allowed",
        "reviewer": "reviewer_01", "reason": "reviewed",
    }])
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_LABEL_OUT_OF_SPACE"


def test_bound_label_removed_from_space_becomes_stale(tmp_path):
    # Bound rows whose label disappears from label_space are stale, not errors,
    # so reviewer/reason/old decision stay traceable.
    config, dataset, path = _bound_base(tmp_path)
    reduced = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
        label_space=["topic_alpha", "topic_beta", "topic_delta", "other"],
    ))
    decisions = _verify(reduced, dataset, path)
    assert len(decisions) == 1
    assert decisions[0].status == "stale"
    assert "ADJ_LABEL_OUT_OF_SPACE" in decisions[0].stale_reasons
    assert "ADJ_SCHEME_CHANGED" in decisions[0].stale_reasons
    items = build_items(dataset, reduced)
    records = {item.item_id: item
               for item in apply_adjudications(items, decisions)}
    record = records["it_1"]
    assert record.source == "adjudication_stale"
    assert record.labels == ()
    assert record.prior_decision_label == "topic_gamma"
    assert record.prior_reviewer == "reviewer_01"
    assert record.adjudicated_by is None


def test_unknown_item_errors(tmp_path):
    config, dataset, _ = _bound_base(tmp_path)
    rows = bind_decisions([
        ("it_1", "topic_alpha", "reviewer_01", "reviewed"),
    ], config, dataset)
    # Rebrand a well-formed bound row to an unknown item.
    rows[0]["item_id"] = "it_404"
    rows[0]["evidence"]["item"]["summary"]["item_id"] = "it_404"
    path = write_jsonl(tmp_path / "unknown.jsonl", rows)
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
    config, dataset, _ = _bound_base(tmp_path)
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(tmp_path / "missing.jsonl", config, dataset.item_ids)
    assert error.value.code == "ADJ_FILE_NOT_FOUND"


# ---------------------------------------------------------------------------
# Evidence binding invalidation scenarios (scheme B: per-item scope)
# ---------------------------------------------------------------------------


def test_unbound_decision_is_stale_and_does_not_override(tmp_path):
    config, dataset, path = _base(tmp_path, [{
        "item_id": "it_1", "decision_label": "topic_gamma",
        "reviewer": "reviewer_01", "reason": "legacy review",
    }])
    decisions = _verify(config, dataset, path)
    assert decisions[0].status == "stale"
    assert decisions[0].stale_reasons == ("ADJ_EVIDENCE_MISSING",)
    assert decisions[0].reviewer == "reviewer_01"
    assert decisions[0].reason == "legacy review"
    assert decisions[0].decision_label == "topic_gamma"
    items = build_items(dataset, config)
    records = {item.item_id: item
               for item in apply_adjudications(items, decisions)}
    # No silent downgrade to majority and no adjudicated override.
    assert records["it_1"].source == "adjudication_stale"
    assert records["it_1"].labels == ()
    assert records["it_1"].prior_decision_label == "topic_gamma"


def test_revision_of_bound_item_invalidates_only_that_item(tmp_path):
    config, dataset, path = _bound_base(
        tmp_path, ("it_1", "topic_gamma", "reviewer_01", "reviewed")
    )
    # Add a second decision for it_2 so we can prove it survives.
    rows = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    second = bind_decisions(
        [("it_2", "topic_beta", "reviewer_02", "unanimous anyway")],
        config, dataset,
    )[0]
    rows.append(json.dumps(second, ensure_ascii=False, sort_keys=True))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    # Revise the raw votes of it_1 only: A03 switches beta -> alpha.
    revised = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
    ])
    decisions = _verify(config, revised, path)
    by_item = {item.item_id: item for item in decisions}
    assert by_item["it_1"].status == "stale"
    assert by_item["it_1"].stale_reasons == ("ADJ_ITEM_EVIDENCE_CHANGED",)
    assert by_item["it_2"].status == "valid"
    items = build_items(revised, config)
    records = {item.item_id: item
               for item in apply_adjudications(items, decisions)}
    assert records["it_1"].source == "adjudication_stale"
    assert records["it_2"].source == "adjudicated"
    assert records["it_2"].labels == ("topic_beta",)


def test_unrelated_item_revision_keeps_other_decisions_valid(tmp_path):
    config, dataset, path = _bound_base(
        tmp_path, ("it_1", "topic_gamma", "reviewer_01", "reviewed")
    )
    # Only it_2 changes; the it_1 decision must stay valid.
    revised = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_alpha"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
    ])
    decisions = _verify(config, revised, path)
    assert len(decisions) == 1
    assert decisions[0].is_valid
    items = build_items(revised, config)
    records = {item.item_id: item
               for item in apply_adjudications(items, decisions)}
    assert records["it_1"].source == "adjudicated"
    assert records["it_1"].labels == ("topic_gamma",)


def test_review_threshold_change_invalidates_bound_decisions(tmp_path):
    config, dataset, path = _bound_base(tmp_path, review_threshold=0.8)
    changed = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.5, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
    ))
    decisions = _verify(changed, dataset, path)
    assert decisions[0].status == "stale"
    assert decisions[0].stale_reasons == ("ADJ_SCHEME_CHANGED",)

def test_label_space_change_invalidates_bound_decisions(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    changed = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
        label_space=LABELS + ["topic_epsilon"],
    ))
    decisions = _verify(changed, dataset, path)
    assert decisions[0].status == "stale"
    assert "ADJ_SCHEME_CHANGED" in decisions[0].stale_reasons


def test_min_annotators_change_invalidates_bound_decisions(tmp_path):
    config, dataset, path = _bound_base(tmp_path, min_annotators=2)
    changed = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=3,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
    ))
    decisions = _verify(changed, dataset, path)
    assert decisions[0].status == "stale"
    assert "ADJ_SCHEME_CHANGED" in decisions[0].stale_reasons


def test_multi_label_toggle_invalidates_bound_decisions(tmp_path):
    config, dataset, path = _bound_base(tmp_path, multi_label=False)
    changed = load_config(make_config(
        tmp_path, metric="pairwise_agreement", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"], multi_label=True,
    ))
    decisions = _verify(changed, dataset, path)
    assert decisions[0].status == "stale"
    assert "ADJ_SCHEME_CHANGED" in decisions[0].stale_reasons


def test_metric_only_change_keeps_binding_valid(tmp_path):
    # Fields outside the decision-affecting set (e.g. the agreement metric)
    # must not invalidate a bound per-item decision.
    config, dataset, path = _bound_base(tmp_path, metric="pairwise_agreement")
    changed = load_config(make_config(
        tmp_path, metric="fleiss", min_annotators=2,
        review_threshold=0.8, adjudication_file=path,
        annotators=["A01", "A02", "A03"],
    ))
    decisions = _verify(changed, dataset, path)
    assert decisions[0].is_valid


def test_tampered_binding_fingerprint_is_detected(tmp_path):
    import json

    config, dataset, path = _bound_base(tmp_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    # Rewrite the embedded vote summary without refreshing the fingerprint.
    rows[0]["evidence"]["item"]["summary"]["votes"][0]["labels"] = ["topic_beta"]
    write_jsonl(path, rows)
    decisions = _verify(config, dataset, path)
    assert decisions[0].status == "stale"
    assert "ADJ_BINDING_TAMPERED" in decisions[0].stale_reasons


def test_missing_annotation_vote_changes_item_fingerprint(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    revised = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", ""),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
    ])
    decisions = _verify(config, revised, path)
    assert decisions[0].status == "stale"
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in decisions[0].stale_reasons


def test_verification_is_deterministic(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    first = _verify(config, dataset, path)
    second = _verify(config, dataset, path)
    assert first == second


def test_malformed_evidence_object_is_a_load_error(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    rows[0]["evidence"] = ["not", "an", "object"]
    write_jsonl(path, rows)
    with pytest.raises(AdjudicationError) as error:
        load_adjudications(path, config, dataset.item_ids)
    assert error.value.code == "ADJ_MALFORMED_ROW"


def test_wrong_binding_scope_is_stale(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    rows[0]["evidence"]["binding_scope"] = "whole_dataset"
    write_jsonl(path, rows)
    decisions = _verify(config, dataset, path)
    assert decisions[0].status == "stale"
    assert decisions[0].stale_reasons == ("ADJ_BINDING_TAMPERED",)


def test_missing_evidence_blocks_is_stale(tmp_path):
    config, dataset, path = _bound_base(tmp_path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    rows[0]["evidence"] = {"binding_schema_version": "1.0",
                           "binding_scope": "per_item"}
    write_jsonl(path, rows)
    decisions = _verify(config, dataset, path)
    assert decisions[0].status == "stale"
    assert "ADJ_BINDING_TAMPERED" in decisions[0].stale_reasons


def test_write_bound_adjudications_roundtrip(tmp_path):
    from tagconsensus.adjudicate import write_bound_adjudications

    config, dataset, path = _bound_base(tmp_path)
    basis = build_evidence_basis(dataset, config)
    loaded = load_adjudications(path, config, dataset.item_ids)
    target = tmp_path / "rebound.jsonl"
    write_bound_adjudications(target, loaded, basis)
    reloaded = load_adjudications(target, config, dataset.item_ids)
    verified = verify_adjudications(reloaded, basis, config)
    assert all(item.is_valid for item in verified)
    # Rebinding is byte-stable.
    target2 = tmp_path / "rebound2.jsonl"
    write_bound_adjudications(target2, loaded, basis)
    assert target.read_text(encoding="utf-8") == target2.read_text(encoding="utf-8")
