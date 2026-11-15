from __future__ import annotations

from tagconsensus.assignment import build_assignment_plan
from tagconsensus.config import load_config
from tagconsensus.validate import validate_dataset

from conftest import make_config, make_dataset, write_manifest_csv


def _finding(result, code):
    return [item for item in result.findings if item.code == code]


def _dataset(tmp_path, rows, name="data.csv"):
    path = tmp_path / name
    lines = ["annotator_id,item_id,label"]
    for annotator, item, label in rows:
        lines.append(f"{annotator},{item},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------- mode selection ------------------------------------------------


def test_full_mode_is_derived_from_whitelist_and_seen_items(tmp_path):
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_beta"),
    ])
    config = load_config(make_config(tmp_path, data, annotators=["A01", "A02", "A03"]))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_beta"),
    ])
    plan = build_assignment_plan(config, dataset)
    assert plan.mode == "full"
    assert plan.pairs == {
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
    }


def test_discovery_mode_does_not_guess_absent_pairs(tmp_path):
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ])
    config = load_config(make_config(tmp_path, data))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ])
    plan = build_assignment_plan(config, dataset)
    assert plan.mode == "discovery"
    assert plan.pairs == frozenset()
    result = validate_dataset(dataset, config, plan)
    assert not _finding(result, "ASSIGNMENT_MISSING")
    assert not _finding(result, "UNASSIGNED_ANNOTATION")
    assert result.coverage["mode"] == "discovery"
    assert result.coverage["coverage_rate"] is None


def test_manifest_mode_uses_manifest_as_authority(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = write_manifest_csv(
        tmp_path / "assign.csv", [("A01", "it_1"), ("A02", "it_1")]
    )
    config = load_config(
        make_config(tmp_path, data, annotators=["A01", "A02"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    plan = build_assignment_plan(config, dataset)
    assert plan.mode == "manifest"
    assert ("A02", "it_1") in plan.pairs
    assert plan.manifest_path.endswith("assign.csv")
    assert len(plan.manifest_sha256) == 64


# ---------- full mode: missing vs empty ----------------------------------


def test_full_mode_flags_missing_assignment_with_identity(tmp_path):
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
    ])
    config = load_config(make_config(tmp_path, data, annotators=["A01", "A02"]))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
    ])
    result = validate_dataset(dataset, config)
    missing = _finding(result, "ASSIGNMENT_MISSING")
    assert len(missing) == 1
    assert missing[0].annotator_id == "A02"
    assert missing[0].item_id == "it_2"
    assert missing[0].severity == "warning"
    assert missing[0].line_no is None


def test_full_mode_distinguishes_explicit_empty_from_missing(tmp_path):
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", ""),
    ])
    config = load_config(make_config(tmp_path, data, annotators=["A01", "A02"]))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", ""),
    ])
    result = validate_dataset(dataset, config)
    assert len(_finding(result, "EMPTY_LABEL")) == 1
    assert not _finding(result, "ASSIGNMENT_MISSING")
    coverage = result.coverage
    assert coverage["submitted_pairs"] == 2
    assert coverage["explicit_empty_pairs"] == 1
    assert coverage["completed_pairs"] == 1
    assert coverage["missing_pairs"] == 0


# ---------- manifest mode: the four-way split ----------------------------


def test_manifest_four_way_split(tmp_path):
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),   # submitted, completed
        ("A02", "it_1", ""),              # explicit empty label
        ("A04", "it_2", "topic_beta"),    # unassigned record
    ])
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"),
        ("A02", "it_1"),
        ("A03", "it_1"),                  # assigned but no record at all
    ])
    config = load_config(
        make_config(tmp_path, data, annotators=["A01", "A02", "A03"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", ""),
        ("A04", "it_2", "topic_beta"),
    ])
    result = validate_dataset(dataset, config)
    missing = _finding(result, "ASSIGNMENT_MISSING")
    assert {(item.annotator_id, item.item_id) for item in missing} == {
        ("A03", "it_1")
    }
    unassigned = _finding(result, "UNASSIGNED_ANNOTATION")
    assert len(unassigned) == 1
    assert unassigned[0].annotator_id == "A04"
    assert unassigned[0].item_id == "it_2"
    assert unassigned[0].severity == "error"
    assert unassigned[0].line_no == 3
    assert len(_finding(result, "EMPTY_LABEL")) == 1
    coverage = result.coverage
    assert coverage["expected_pairs"] == 3
    assert coverage["submitted_pairs"] == 2
    assert coverage["explicit_empty_pairs"] == 1
    assert coverage["completed_pairs"] == 1
    assert coverage["missing_pairs"] == 1
    assert coverage["unassigned_records"] == 1
    assert result.blocked is True  # UNASSIGNED_ANNOTATION is an error


def test_manifest_item_outside_scope_does_not_falsely_report_missing(tmp_path):
    # it_2 exists in the data but is outside the manifest's declared scope.
    data = _dataset(tmp_path, [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
    ])
    manifest = write_manifest_csv(
        tmp_path / "assign.csv", [("A01", "it_1"), ("A02", "it_1")]
    )
    config = load_config(
        make_config(tmp_path, data, annotators=["A01", "A02"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
    ])
    result = validate_dataset(dataset, config)
    # it_2/A01 is out of scope -> unassigned, never a missing assignment.
    missing_items = {item.item_id for item in _finding(result, "ASSIGNMENT_MISSING")}
    assert "it_2" not in missing_items
    unassigned = _finding(result, "UNASSIGNED_ANNOTATION")
    assert {(item.annotator_id, item.item_id) for item in unassigned} == {
        ("A01", "it_2")
    }
    assert result.coverage["missing_pairs"] == 0
    assert result.coverage["unassigned_records"] == 1


def test_manifest_does_not_invent_items_absent_from_data(tmp_path):
    # Manifest declares it_9, which never appears in the dataset; that is a
    # genuine missing assignment, not a phantom item.
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = write_manifest_csv(
        tmp_path / "assign.csv", [("A01", "it_1"), ("A01", "it_9")]
    )
    config = load_config(
        make_config(tmp_path, data, annotators=["A01"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    missing = _finding(result, "ASSIGNMENT_MISSING")
    assert {(item.annotator_id, item.item_id) for item in missing} == {
        ("A01", "it_9")
    }
    assert result.coverage["expected_pairs"] == 2


