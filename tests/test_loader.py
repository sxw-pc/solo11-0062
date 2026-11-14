from __future__ import annotations

import pytest

from tagconsensus.config import load_config
from tagconsensus.errors import DataError
from tagconsensus.loader import load_dataset, normalize_labels

from conftest import make_config, write_jsonl


def _load(tmp_path, path, **overrides):
    config = load_config(make_config(tmp_path, path, **overrides))
    return load_dataset(config.dataset)


def test_csv_normalizes_whitespace_and_case(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "annotator_id,item_id,label,confidence\n"
        " A01 , it_1 , TOPIC_ALPHA ,0.8\n"
        "A02,it_1,  topic_beta  ,\n",
        encoding="utf-8",
    )
    dataset = _load(tmp_path, path)
    assert [row.labels for row in dataset.annotations] == [
        ("topic_alpha",), ("topic_beta",)
    ]
    assert dataset.annotations[0].annotator_id == "A01"
    assert dataset.annotations[0].item_id == "it_1"
    assert dataset.annotations[0].confidence == 0.8
    assert dataset.annotations[1].confidence is None
    assert dataset.records == 2


def test_empty_label_is_kept_as_missing(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "annotator_id,item_id,label\nA01,it_1,\nA02,it_1,topic_alpha\n",
        encoding="utf-8",
    )
    dataset = _load(tmp_path, path)
    assert dataset.missing_annotations == 1
    assert dataset.annotations[0].is_missing


def test_csv_utf8_sig(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_bytes(
        "﻿annotator_id,item_id,label\nA01,it_1,topic_alpha\n".encode("utf-8")
    )
    dataset = _load(tmp_path, path, extra={"dataset": {
        "path": str(path), "format": "csv", "encoding": "utf-8-sig"}})
    assert dataset.records == 1
    assert dataset.annotations[0].labels == ("topic_alpha",)


def test_csv_gbk_encoding(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_bytes("annotator_id,item_id,label\nA01,it_1,topic_beta\n".encode("gbk"))
    config = load_config(make_config(
        tmp_path, path,
        extra={"dataset": {"path": str(path), "format": "csv", "encoding": "gbk"}},
    ))
    dataset = load_dataset(config.dataset)
    assert dataset.annotations[0].labels == ("topic_beta",)


def test_bad_encoding_raises_stable_error(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_bytes(b"annotator_id,item_id,label\nA01,it_1,\xff\n")
    config = load_config(make_config(
        tmp_path, path,
        extra={"dataset": {"path": str(path), "format": "csv", "encoding": "utf-8"}},
    ))
    with pytest.raises(DataError) as error:
        load_dataset(config.dataset)
    assert error.value.code == "DATA_ENCODING"


def test_jsonl_string_and_array_labels(tmp_path):
    path = tmp_path / "annotations.jsonl"
    write_jsonl(path, [
        {"annotator_id": "A01", "item_id": "it_1", "label": ["Topic_Beta", "topic_alpha"]},
        {"annotator_id": "A02", "item_id": "it_1", "label": "topic_alpha"},
        {"annotator_id": "A03", "item_id": "it_1", "label": []},
    ])
    config = load_config(make_config(
        tmp_path, path,
        multi_label=True,
        extra={"dataset": {"path": str(path), "format": "jsonl", "encoding": "utf-8"}},
    ))
    dataset = load_dataset(config.dataset)
    assert dataset.annotations[0].labels == ("topic_alpha", "topic_beta")
    assert dataset.annotations[1].labels == ("topic_alpha",)
    assert dataset.annotations[2].is_missing


def test_malformed_rows_report_exact_line_numbers(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\n"
        "A02,it_1\n"
        ",it_2,topic_beta\n"
        "A03,it_2,topic_gamma\n",
        encoding="utf-8",
    )
    dataset = _load(tmp_path, path)
    assert dataset.records == 2
    malformed = [item for item in dataset.parse_findings if item.code == "MALFORMED_ROW"]
    assert [item.line_no for item in malformed] == [3, 4]


def test_jsonl_bad_json_line_number(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        '{"annotator_id":"A01","item_id":"it_1","label":"topic_alpha"}\n'
        '{"annotator_id":\n',
        encoding="utf-8",
    )
    config = load_config(make_config(
        tmp_path, path,
        extra={"dataset": {"path": str(path), "format": "jsonl", "encoding": "utf-8"}},
    ))
    dataset = load_dataset(config.dataset)
    assert dataset.records == 1
    assert dataset.parse_findings[0].line_no == 2
    assert dataset.parse_findings[0].code == "MALFORMED_ROW"


def test_empty_file_and_header_only_are_zero_records(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    assert _load(tmp_path, empty).records == 0
    header = tmp_path / "header.csv"
    header.write_text("annotator_id,item_id,label\n", encoding="utf-8")
    assert _load(tmp_path, header).records == 0


def test_missing_header_column_raises(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text("annotator_id,item_id\nA01,it_1\n", encoding="utf-8")
    with pytest.raises(DataError) as error:
        _load(tmp_path, path)
    assert error.value.code == "DATA_HEADER"


def test_missing_file_raises(tmp_path):
    path = tmp_path / "absent.csv"
    config = load_config(make_config(tmp_path, path))
    path.unlink()
    with pytest.raises(DataError) as error:
        load_dataset(config.dataset)
    assert error.value.code == "DATA_FILE_NOT_FOUND"


def test_fingerprint_contains_hash_count_and_annotators(tmp_path):
    path = tmp_path / "annotations.csv"
    path.write_text(
        "annotator_id,item_id,label\nA02,it_1,topic_alpha\nA01,it_1,topic_beta\n",
        encoding="utf-8",
    )
    dataset = _load(tmp_path, path)
    fingerprint = dataset.fingerprint_dict()
    assert fingerprint["record_count"] == 2
    assert len(fingerprint["file_sha256"]) == 64
    assert len(fingerprint["annotator_fingerprint"]) == 64
    assert dataset.annotators == ("A01", "A02")


def test_normalize_labels_helpers():
    assert normalize_labels([" B ", "a", "b"]) == ("a", "b")
    assert normalize_labels("") == ()
    with pytest.raises(ValueError):
        normalize_labels(3)  # type: ignore[arg-type]
