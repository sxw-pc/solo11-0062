from __future__ import annotations

from tagconsensus.config import load_config
from tagconsensus.models import Dataset, ValidationFinding
from tagconsensus.validate import validate_dataset

from conftest import make_config, make_dataset


def test_duplicate_annotation_is_an_error_with_line(tmp_path):
    config = load_config(make_config(tmp_path))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A01", "it_1", "topic_beta"),
    ])
    result = validate_dataset(dataset, config)
    duplicates = [item for item in result.findings if item.code == "DUPLICATE_ANNOTATION"]
    assert duplicates
    assert duplicates[0].line_no == 2
    assert result.blocked is True


def test_out_of_space_unknown_and_insufficient(tmp_path):
    config = load_config(make_config(tmp_path, annotators=["A01"], min_annotators=3))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A99", "it_1", "not_a_label"),
    ])
    result = validate_dataset(dataset, config)
    codes = {item.code for item in result.findings}
    assert {"UNKNOWN_ANNOTATOR", "LABEL_OUT_OF_SPACE",
            "INSUFFICIENT_ANNOTATORS"} <= codes
    unknown = next(item for item in result.findings if item.code == "UNKNOWN_ANNOTATOR")
    assert unknown.annotator_id == "A99"
    assert unknown.line_no == 2


def test_empty_label_warning_does_not_block_non_strict(tmp_path):
    config = load_config(make_config(tmp_path, min_annotators=2))
    dataset = make_dataset([
        ("A01", "it_1", ""),
        ("A02", "it_1", "topic_alpha"),
    ])
    result = validate_dataset(dataset, config)
    assert result.error_count == 0
    assert result.warning_count == 2
    assert result.status == "warn"
    assert any(item.code == "EMPTY_LABEL" for item in result.findings)


def test_strict_promotes_warning_to_blocking_error(tmp_path):
    config = load_config(make_config(tmp_path, min_annotators=2, strict=True))
    dataset = make_dataset([
        ("A01", "it_1", ""),
        ("A02", "it_1", "topic_alpha"),
    ])
    result = validate_dataset(dataset, config)
    assert result.blocked is True
    assert result.warning_count == 0
    assert result.promoted_count >= 1
    empty = [item for item in result.findings if item.code == "EMPTY_LABEL"]
    assert empty and empty[0].severity == "error"
    assert empty[0].promoted is True


def test_clean_dataset_passes(tmp_path):
    config = load_config(make_config(tmp_path, min_annotators=2))
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
    ])
    result = validate_dataset(dataset, config)
    assert result.error_count == 0
    assert result.warning_count == 0
    assert result.status == "pass"


def test_multi_label_row_requires_multi_label_mode(tmp_path):
    config = load_config(make_config(tmp_path, multi_label=False))
    dataset = make_dataset([
        ("A01", "it_1", ["topic_alpha", "topic_beta"]),
        ("A02", "it_1", "topic_alpha"),
    ])
    result = validate_dataset(dataset, config)
    assert any(item.code == "MULTI_LABEL_NOT_ENABLED" for item in result.findings)


def test_parse_findings_are_included_and_empty_dataset_blocks(tmp_path):
    config = load_config(make_config(tmp_path))
    parse = ValidationFinding("MALFORMED_ROW", "error", "bad row", line_no=4)
    dataset = Dataset((), "x", "csv", "sha", (parse,))
    result = validate_dataset(dataset, config)
    codes = [item.code for item in result.findings]
    assert "MALFORMED_ROW" in codes
    assert "DATA_EMPTY" in codes


def test_findings_are_sorted_deterministically(tmp_path):
    config = load_config(make_config(tmp_path, annotators=["A01"]))
    dataset = make_dataset([
        ("A99", "it_2", "bad"),
        ("A98", "it_1", "bad"),
    ])
    first = validate_dataset(dataset, config).findings
    second = validate_dataset(dataset, config).findings
    assert first == second
    severities = [item.severity for item in first]
    assert severities == sorted(severities, key=lambda value: {"error": 0}.get(value, 1))
