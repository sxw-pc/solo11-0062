from __future__ import annotations

import pytest

from tagconsensus.assignment import build_assignment_plan
from tagconsensus.config import load_config
from tagconsensus.errors import DataError
from tagconsensus.validate import validate_dataset

from conftest import make_config, make_dataset


def _data(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label\nA01,it_1,topic_alpha\n", encoding="utf-8"
    )
    return path


def test_empty_manifest_produces_no_pairs(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.csv"
    manifest.write_text("annotator_id,item_id\n", encoding="utf-8")
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    assert result.coverage["expected_pairs"] == 0
    assert result.coverage["coverage_rate"] is None
    unassigned = [
        item for item in result.findings if item.code == "UNASSIGNED_ANNOTATION"
    ]
    assert len(unassigned) == 1


def test_manifest_blank_lines_are_skipped(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.csv"
    manifest.write_text(
        "annotator_id,item_id\n\nA01,it_1\n\n", encoding="utf-8"
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    assert result.coverage["expected_pairs"] == 1
    assert not [
        item for item in result.findings
        if item.code == "ASSIGNMENT_MALFORMED_ROW"
    ]


def test_manifest_gbk_encoding_roundtrip(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign_gbk.csv"
    manifest.write_bytes(
        "annotator_id,item_id\n标注员1,it_1\n".encode("gbk")
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["标注员1"],
        assignment_manifest={"path": str(manifest), "encoding": "gbk"},
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    plan = build_assignment_plan(config, dataset)
    assert plan.pairs == {("标注员1", "it_1")}


def test_manifest_encoding_error_is_located(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.csv"
    # GBK bytes for a CJK annotator id, which are not valid utf-8.
    manifest.write_bytes(
        "annotator_id,item_id\n".encode("utf-8") + "标注员1,it_1\n".encode("gbk")
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"],
        assignment_manifest={"path": str(manifest), "encoding": "utf-8"},
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    with pytest.raises(DataError) as error:
        build_assignment_plan(config, dataset)
    assert error.value.code == "ASSIGNMENT_ENCODING"
    assert error.value.field == "assignment_manifest.encoding"


def test_duplicate_header_raises_data_error(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.csv"
    manifest.write_text(
        "annotator_id,annotator_id,item_id\nA01,A01,it_1\n", encoding="utf-8"
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    with pytest.raises(DataError) as error:
        build_assignment_plan(config, dataset)
    assert error.value.code == "ASSIGNMENT_HEADER"


def test_jsonl_manifest_with_blank_lines(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.jsonl"
    manifest.write_text(
        '\n{"annotator_id": "A01", "item_id": "it_1"}\n\n', encoding="utf-8"
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    assert result.coverage["expected_pairs"] == 1
    assert result.blocked is False


def test_empty_csv_manifest_file_is_allowed(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "empty.csv"
    manifest.write_text("", encoding="utf-8")
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    assert result.coverage["expected_pairs"] == 0
    assert len([
        item for item in result.findings if item.code == "UNASSIGNED_ANNOTATION"
    ]) == 1


def test_jsonl_invalid_json_is_a_located_finding(tmp_path):
    data = _data(tmp_path)
    manifest = tmp_path / "assign.jsonl"
    manifest.write_text(
        '{"annotator_id": "A01", "item_id": "it_1"}\n{not json}\n',
        encoding="utf-8",
    )
    config = load_config(make_config(
        tmp_path, data, annotators=["A01"], assignment_manifest=manifest,
    ))
    dataset = make_dataset([("A01", "it_1", "topic_alpha")])
    result = validate_dataset(dataset, config)
    malformed = [
        item for item in result.findings
        if item.code == "ASSIGNMENT_MALFORMED_ROW"
    ]
    assert len(malformed) == 1
    assert malformed[0].line_no == 2
    assert result.blocked is True
