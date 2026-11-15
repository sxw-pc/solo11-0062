from __future__ import annotations

import pytest

from tagconsensus.assignment import build_assignment_plan
from tagconsensus.config import load_config
from tagconsensus.errors import DataError
from tagconsensus.validate import validate_dataset

from conftest import make_config, make_dataset, write_manifest_csv, write_jsonl


def _finding(result, code):
    return [item for item in result.findings if item.code == code]


def _dataset(tmp_path, rows, name="data.csv"):
    path = tmp_path / name
    lines = ["annotator_id,item_id,label"]
    for annotator, item, label in rows:
        lines.append(f"{annotator},{item},{label}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------- manifest integrity errors ------------------------------------


def test_duplicate_manifest_pair_is_error_with_line(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A01", "it_1"),
    ])
    config = load_config(
        make_config(tmp_path, data, annotators=["A01"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    duplicates = _finding(result, "ASSIGNMENT_DUPLICATE_PAIR")
    assert len(duplicates) == 1
    assert duplicates[0].line_no == 3
    assert duplicates[0].severity == "error"
    assert duplicates[0].annotator_id == "A01"
    assert result.blocked is True
    # The duplicate pair is still counted once as an expected assignment.
    assert result.coverage["expected_pairs"] == 1


def test_manifest_annotator_outside_whitelist_conflicts(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = write_manifest_csv(
        tmp_path / "assign.csv", [("A01", "it_1"), ("A99", "it_1")]
    )
    config = load_config(
        make_config(tmp_path, data, annotators=["A01"],
                    assignment_manifest=manifest)
    )
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    unknown = _finding(result, "ASSIGNMENT_UNKNOWN_ANNOTATOR")
    assert len(unknown) == 1
    assert unknown[0].annotator_id == "A99"
    assert unknown[0].item_id == "it_1"
    assert unknown[0].line_no == 3
    assert result.blocked is True


def test_manifest_malformed_row_and_jsonl(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")], name="d.csv")
    csv_manifest = tmp_path / "assign.csv"
    csv_manifest.write_text(
        "annotator_id,item_id\nA01,it_1\n,it_1\nA02,\n", encoding="utf-8"
    )
    config = load_config(
        make_config(tmp_path, data, annotators=["A01", "A02"],
                    assignment_manifest=csv_manifest)
    )
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    malformed = _finding(result, "ASSIGNMENT_MALFORMED_ROW")
    assert {item.line_no for item in malformed} == {3, 4}

    jsonl_manifest = write_jsonl(tmp_path / "assign.jsonl", [
        {"annotator_id": "A01", "item_id": "it_1"},
        {"annotator_id": "", "item_id": "it_1"},
        "not-an-object",
    ])
    config_jsonl = load_config(
        make_config(tmp_path, data, annotators=["A01"],
                    assignment_manifest=jsonl_manifest)
    )
    result_jsonl = validate_dataset(dataset, config_jsonl)
    assert len(_finding(result_jsonl, "ASSIGNMENT_MALFORMED_ROW")) == 2


def test_manifest_missing_file_and_header_errors(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    missing_config = load_config(make_config(
        tmp_path, data, annotators=["A01"],
        assignment_manifest=tmp_path / "nope.csv",
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    with pytest.raises(DataError) as error:
        build_assignment_plan(missing_config, dataset)
    assert error.value.code == "ASSIGNMENT_FILE_NOT_FOUND"

    bad_header = tmp_path / "bad.csv"
    bad_header.write_text("annotator_id,label\nA01,x\n", encoding="utf-8")
    bad_config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=bad_header,
    ))
    with pytest.raises(DataError) as error:
        build_assignment_plan(bad_config, dataset)
    assert error.value.code == "ASSIGNMENT_HEADER"


# ---------- config parsing ------------------------------------------------


def test_manifest_config_accepts_string_and_mapping(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = write_manifest_csv(
        tmp_path / "assign.jsonl", [("A01", "it_1")]
    )
    string_config = load_config(
        make_config(tmp_path, data, assignment_manifest=manifest)
    )
    assert string_config.assignment_manifest.format == "jsonl"
    mapping_config = load_config(make_config(
        tmp_path, data,
        assignment_manifest={"path": str(manifest), "format": "jsonl"},
    ))
    assert mapping_config.assignment_manifest.format == "jsonl"
    assert mapping_config.assignment_manifest.path == manifest.resolve()


def test_manifest_config_rejects_bad_format_encoding_and_field(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    manifest = tmp_path / "assign.txt"
    manifest.write_text("x\n", encoding="utf-8")
    with pytest.raises(Exception) as error:
        load_config(make_config(tmp_path, data, assignment_manifest=manifest))
    assert error.value.code == "CFG_ASSIGNMENT_FORMAT"
    with pytest.raises(Exception) as error:
        load_config(make_config(
            tmp_path, data,
            assignment_manifest={"path": "a.csv", "encoding": "latin-1"},
        ))
    assert error.value.code == "CFG_ENCODING_UNKNOWN"
    with pytest.raises(Exception) as error:
        load_config(make_config(
            tmp_path, data,
            assignment_manifest={"path": "a.csv", "bogus": True},
        ))
    assert error.value.code == "CFG_UNKNOWN_FIELD"


def test_manifest_changes_config_fingerprint(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    first = load_config(make_config(tmp_path, data, annotators=["A01"]))
    manifest = write_manifest_csv(
        tmp_path / "assign.csv", [("A01", "it_1")]
    )
    second = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    assert first.fingerprint != second.fingerprint
    summary = second.summary()
    assert summary["assignment_manifest"]["path"].endswith("assign.csv")


def test_manifest_config_rejects_non_mapping_and_empty_path(tmp_path):
    data = _dataset(tmp_path, [("A01", "it_1", "topic_alpha")])
    with pytest.raises(Exception) as error:
        load_config(make_config(
            tmp_path, data, assignment_manifest=["not", "a", "mapping"],
        ))
    assert error.value.code == "CFG_TYPE_INVALID"
    with pytest.raises(Exception) as error:
        load_config(make_config(
            tmp_path, data,
            assignment_manifest={"path": "  "},
        ))
    assert error.value.code == "CFG_MISSING_FIELD"
    assert error.value.field == "assignment_manifest.path"
