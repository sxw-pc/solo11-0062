"""Bounded-memory staging path: equivalence, diagnostics and failure semantics.

Every correctness test compares the SQLite-backed staged pipeline against the
legacy in-memory loader so reports stay item-identical, while the memory tests
prove the input file is never read whole regardless of how large it grows.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest
import yaml

from tagconsensus.config import DatasetSpec, load_config
from tagconsensus.errors import DataError
from tagconsensus.items import build_items, leave_one_out
from tagconsensus.loader import load_dataset, stream_text_lines
from tagconsensus.metrics import compute_metric
from tagconsensus.staging import stage_dataset
from tagconsensus.validate import validate_dataset

from conftest import make_config


def _spec(path: Path, fmt: str = "csv", encoding: str = "utf-8") -> DatasetSpec:
    return DatasetSpec(path, str(path), fmt, encoding)


def _rows(source) -> list[tuple]:
    return [
        (row.annotator_id, row.item_id, row.labels, row.confidence, row.line_no)
        for row in source.iter_annotations()
    ]


def _findings(source) -> list[dict]:
    return [item.to_dict() for item in source.iter_parse_findings()]


# --------------------------------------------------------------------------
# Equivalence with the in-memory path
# --------------------------------------------------------------------------


def test_staged_csv_matches_in_memory_on_large_synthetic_input(tmp_path):
    path = tmp_path / "large.csv"
    lines = ["annotator_id,item_id,label,confidence"]
    # More rows than the insert batch size, with repeated item/annotator keys.
    for index in range(3200):
        annotator = f"A{index % 7:02d}"
        item = f"it_{index % 53:04d}"
        label = "topic_alpha" if index % 2 else "topic_beta"
        confidence = "" if index % 5 == 0 else f"0.{index % 10}"
        lines.append(f"{annotator},{item},{label},{confidence}")
    # Quoted embedded newline keeps the physical line numbering honest.
    lines.insert(1500, 'A06,it_quoted,"topic\ngamma",0.4')
    # Empty label rows, malformed width and a bad confidence.
    lines.insert(2200, "A02,it_empty,,0.9")
    lines.insert(2201, "A03,it_bad,topic_alpha,1.5")
    lines.insert(2202, "A04,too_few")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spec = _spec(path)
    memory = load_dataset(spec)
    # A chunk smaller than any individual record forces many buffer flips.
    with stage_dataset(spec, chunk_size=32) as staged:
        assert staged.records == memory.records
        assert _rows(staged) == _rows(memory)
        assert _findings(staged) == _findings(memory)
        assert list(staged.iter_duplicate_groups()) == list(
            memory.iter_duplicate_groups()
        )
        assert staged.file_sha256 == memory.file_sha256
        assert staged.item_ids == memory.item_ids
        assert staged.annotators == memory.annotators
        assert staged.missing_annotations == memory.missing_annotations
        assert staged.fingerprint_dict() == memory.fingerprint_dict()


def test_staged_jsonl_matches_in_memory(tmp_path):
    path = tmp_path / "large.jsonl"
    lines = []
    for index in range(2500):
        lines.append(
            json.dumps(
                {
                    "annotator_id": f"A{index % 5:02d}",
                    "item_id": f"it_{index % 41:04d}",
                    "label": ["topic_alpha", "topic_beta"] if index % 3 == 0 else "topic_alpha",
                    "confidence": 0.5,
                },
                sort_keys=True,
            )
        )
    lines.insert(800, "")  # blank physical line
    lines.insert(801, "{not valid json")
    lines.insert(802, json.dumps({"annotator_id": "A01", "item_id": "it_x"}))  # no label
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spec = _spec(path, fmt="jsonl")
    memory = load_dataset(spec)
    with stage_dataset(spec, chunk_size=17) as staged:
        assert _rows(staged) == _rows(memory)
        assert _findings(staged) == _findings(memory)
        assert list(staged.iter_duplicate_groups()) == list(
            memory.iter_duplicate_groups()
        )
        assert staged.file_sha256 == memory.file_sha256


def test_quoted_newline_line_numbers_identical_across_chunk_sizes(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\n"
        "A02,\"it\n_2\",topic_beta\n"
        "A03,it_3,\n"
        "A04,it_4,topic_gamma\n",
        encoding="utf-8",
    )
    spec = _spec(path)
    memory = load_dataset(spec)
    expected_lines = [row.line_no for row in memory.iter_annotations()]
    for chunk_size in (1, 2, 5, 11, 64, 4096):
        with stage_dataset(spec, chunk_size=chunk_size) as staged:
            assert [row.line_no for row in staged.iter_annotations()] == expected_lines
            assert _rows(staged) == _rows(memory)


def test_full_pipeline_reports_are_identical(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_alpha\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_alpha\n"
        "A01,it_3,\nA02,it_3,topic_beta\nA03,it_3,topic_beta\n",
        encoding="utf-8",
    )
    config = load_config(
        make_config(
            tmp_path,
            path,
            metric="pairwise_agreement",
            annotators=["A01", "A02", "A03"],
        )
    )
    memory = load_dataset(config.dataset)
    with stage_dataset(config.dataset) as staged:
        assert validate_dataset(staged, config).to_dict() == validate_dataset(
            memory, config
        ).to_dict()
        assert compute_metric(staged, config).to_dict() == compute_metric(
            memory, config
        ).to_dict()
        assert [item.to_dict() for item in build_items(staged, config)] == [
            item.to_dict() for item in build_items(memory, config)
        ]
        baseline_staged = compute_metric(staged, config)
        baseline_memory = compute_metric(memory, config)
        assert [item.to_dict() for item in leave_one_out(staged, config, baseline_staged)] == [
            item.to_dict() for item in leave_one_out(memory, config, baseline_memory)
        ]


# --------------------------------------------------------------------------
# Global duplicate detection across chunks
# --------------------------------------------------------------------------


def test_duplicate_keys_detected_when_repeated_many_batches_apart(tmp_path):
    path = tmp_path / "dups.csv"
    lines = ["annotator_id,item_id,label"]
    for index in range(2500):
        lines.append(f"A{index % 9:02d},it_{index:05d},topic_alpha")
    # Repeat a key from the very beginning at the end of the file.
    lines.append("A00,it_00000,topic_beta")
    lines.append("A00,it_00000,topic_gamma")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    spec = _spec(path)

    with stage_dataset(spec, chunk_size=64) as staged:
        groups = dict(staged.iter_duplicate_groups())
    assert ("A00", "it_00000") in groups
    first_line, second_line, third_line = groups[("A00", "it_00000")]
    assert first_line == 2
    assert second_line == 2502
    assert third_line == 2503

    config = load_config(make_config(tmp_path, path))
    memory = load_dataset(spec)
    with stage_dataset(spec, chunk_size=64) as staged:
        staged_findings = validate_dataset(staged, config).to_dict()
    memory_findings = validate_dataset(memory, config).to_dict()
    assert staged_findings == memory_findings
    duplicate_lines = [
        item["line_no"]
        for item in staged_findings["findings"]
        if item["code"] == "DUPLICATE_ANNOTATION"
    ]
    assert duplicate_lines == [2502, 2503]
    messages = [
        item["message"]
        for item in staged_findings["findings"]
        if item["code"] == "DUPLICATE_ANNOTATION"
    ]
    assert all("first at line 2" in message for message in messages)


# --------------------------------------------------------------------------
# Bounded memory: the input file is never read whole
# --------------------------------------------------------------------------


def test_streaming_never_reads_more_than_the_chunk_size(tmp_path, monkeypatch):
    path = tmp_path / "bounded.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        + "".join(f"A0{i % 3},it_{i},topic_alpha\n" for i in range(500)),
        encoding="utf-8",
    )
    chunk_size = 128
    seen_sizes: list[int] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        if self == path:
            original_read_method = handle.read

            def tracking_read(size=-1):
                if size is not None and size > 0:
                    seen_sizes.append(size)
                    assert size <= chunk_size
                return original_read_method(size)

            handle.read = tracking_read  # type: ignore[method-assign]
        return handle

    monkeypatch.setattr(Path, "open", tracking_open)
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(
            AssertionError("staged path must not call read_bytes()")
        ),
    )
    spec = _spec(path)
    with stage_dataset(spec, chunk_size=chunk_size) as staged:
        assert staged.records == 500
    assert seen_sizes, "expected chunked reads of the input file"
    assert max(seen_sizes) <= chunk_size


def test_stream_text_lines_matches_splitlines_at_every_boundary(tmp_path):
    payload = (
        "a\nb\r\nc\rd\ve\f\x1c\x1d\x1e\x85 x y"
        "中文,quoted\r\n tail"
    ).encode("utf-8")
    path = tmp_path / "terms.txt"
    path.write_bytes(payload)
    expected = payload.decode("utf-8").splitlines(keepends=True)
    for chunk_size in range(1, 24):
        hasher = hashlib.sha256()
        lines = [
            line for _, line in stream_text_lines(path, "utf-8", hasher, chunk_size)
        ]
        assert lines == expected
        assert hasher.hexdigest() == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------------


def test_gbk_and_utf8_sig_equivalence_with_split_multibyte_chars(tmp_path):
    gbk_path = tmp_path / "g.csv"
    gbk_path.write_bytes(
        "annotator_id,item_id,label\nA01,it_1,中文标签\nA02,it_2,other\n".encode("gbk")
    )
    gbk_spec = _spec(gbk_path, encoding="gbk")
    gbk_memory = load_dataset(gbk_spec)
    with stage_dataset(gbk_spec, chunk_size=3) as staged:
        assert _rows(staged) == _rows(gbk_memory)
        assert staged.file_sha256 == gbk_memory.file_sha256

    sig_path = tmp_path / "sig.csv"
    sig_path.write_bytes(
        "﻿annotator_id,item_id,label\nA01,it_1,topic_alpha\n".encode("utf-8")
    )
    sig_spec = _spec(sig_path, encoding="utf-8-sig")
    sig_memory = load_dataset(sig_spec)
    with stage_dataset(sig_spec, chunk_size=2) as staged:
        assert staged.records == sig_memory.records == 1
        assert _rows(staged) == _rows(sig_memory)


def test_mid_stream_decoding_failure_raises_encoding_error_and_cleans_up(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_bytes(
        b"annotator_id,item_id,label\n"
        b"A01,it_1,topic_alpha\n"
        b"A02,it_2,\xff\n"
    )
    spec = _spec(path)
    with pytest.raises(DataError) as error:
        stage_dataset(spec, chunk_size=8)
    assert error.value.code == "DATA_ENCODING"
    leftovers = [
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("tagconsensus-stage-")
    ]
    assert leftovers == []
    # The user's input file must remain untouched.
    assert path.exists()
    assert path.read_bytes().startswith(b"annotator_id")


# --------------------------------------------------------------------------
# Staging failures: rollback, no partial dataset, safe cleanup
# --------------------------------------------------------------------------


def test_staging_write_failure_rolls_back_and_removes_private_temp_dir(
    tmp_path, monkeypatch
):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        + "".join(f"A0{i % 4},it_{i},topic_alpha\n" for i in range(100)),
        encoding="utf-8",
    )
    sentinel = tmp_path / "user_data.txt"
    sentinel.write_text("must not be deleted", encoding="utf-8")

    real_connect = sqlite3.connect

    class FailingConnection:
        """Wrap a real connection and fail the first bulk insert once."""

        def __init__(self, raw):
            self._raw = raw
            self._failed = False

        def executemany(self, statement, parameters):
            if not self._failed:
                self._failed = True
                raise sqlite3.OperationalError("simulated disk failure")
            return self._raw.executemany(statement, parameters)

        def executescript(self, script):
            return self._raw.executescript(script)

        def execute(self, *args, **kwargs):
            return self._raw.execute(*args, **kwargs)

        def rollback(self):
            return self._raw.rollback()

        def commit(self):
            return self._raw.commit()

        def close(self):
            return self._raw.close()

    def failing_connect(database, *args, **kwargs):
        raw = real_connect(database, *args, **kwargs)
        if str(database).endswith("stage.db"):
            return FailingConnection(raw)
        return raw

    monkeypatch.setattr(sqlite3, "connect", failing_connect)
    with pytest.raises(DataError) as error:
        stage_dataset(_spec(path))
    assert error.value.code == "DATA_STAGING"

    stage_dirs = [
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("tagconsensus-stage-")
    ]
    assert stage_dirs == []
    # Only the private stage directory is ever removed; user data is preserved.
    assert sentinel.read_text(encoding="utf-8") == "must not be deleted"
    assert path.exists()


def test_malformed_rows_block_and_keep_full_diagnostics(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\n"
        "A02,it_1\n"
        ",it_2,topic_beta\n",
        encoding="utf-8",
    )
    config = load_config(make_config(tmp_path, path))
    memory = load_dataset(_spec(path))
    with stage_dataset(_spec(path)) as staged:
        result = validate_dataset(staged, config)
        assert result.blocked is True
        assert result.to_dict() == validate_dataset(memory, config).to_dict()
        line_numbers = [
            item.line_no
            for item in result.findings
            if item.code == "MALFORMED_ROW"
        ]
        assert line_numbers == [3, 4]
        # The valid prefix is still available for diagnostics but never reaches
        # metrics through the CLI, which only computes them when not blocked.
        assert staged.records == 1


def test_empty_and_header_only_files_stage_as_zero_records(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    header = tmp_path / "header.csv"
    header.write_text("annotator_id,item_id,label\n", encoding="utf-8")
    for path in (empty, header):
        memory = load_dataset(_spec(path))
        with stage_dataset(_spec(path)) as staged:
            assert staged.records == 0
            assert _rows(staged) == _rows(memory) == []
            assert staged.file_sha256 == memory.file_sha256
            config = load_config(make_config(tmp_path, path))
            result = validate_dataset(staged, config)
            assert any(item.code == "DATA_EMPTY" for item in result.findings)
            assert result.blocked is True


def test_header_errors_raise_and_clean_up(tmp_path):
    path = tmp_path / "bad_header.csv"
    path.write_text("annotator_id,item_id\nA01,it_1\n", encoding="utf-8")
    with pytest.raises(DataError) as error:
        stage_dataset(_spec(path))
    assert error.value.code == "DATA_HEADER"
    assert not [
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("tagconsensus-stage-")
    ]


def test_missing_file_raises_stable_error_without_leaving_temp_dir(tmp_path):
    spec = _spec(tmp_path / "absent.csv")
    with pytest.raises(DataError) as error:
        stage_dataset(spec)
    assert error.value.code == "DATA_FILE_NOT_FOUND"
    assert not [
        name
        for name in os.listdir(tempfile.gettempdir())
        if name.startswith("tagconsensus-stage-")
    ]


# --------------------------------------------------------------------------
# Filtered views and lifecycle
# --------------------------------------------------------------------------


def test_without_annotator_view_matches_in_memory_dataset(tmp_path):
    path = tmp_path / "view.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_beta\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_beta\n"
        "A01,it_1,topic_gamma\n",
        encoding="utf-8",
    )
    spec = _spec(path)
    memory = load_dataset(spec)
    with stage_dataset(spec) as staged:
        view = staged.without_annotator("A01")
        memory_view = memory.without_annotator("A01")
        assert _rows(view) == _rows(memory_view)
        assert view.item_ids == memory_view.item_ids
        assert view.annotators == memory_view.annotators
        assert view.missing_annotations == memory_view.missing_annotations
        assert list(view.iter_duplicate_groups()) == list(
            memory_view.iter_duplicate_groups()
        )
        config = load_config(
            make_config(tmp_path, path, annotators=["A01", "A02", "A03"])
        )
        assert compute_metric(view, config).to_dict() == compute_metric(
            memory_view, config
        ).to_dict()


def test_context_manager_removes_temp_dir_on_normal_exit(tmp_path):
    path = tmp_path / "clean.csv"
    path.write_text(
        "annotator_id,item_id,label\nA01,it_1,topic_alpha\nA02,it_1,topic_alpha\n",
        encoding="utf-8",
    )
    with stage_dataset(_spec(path)) as staged:
        temp_dir = staged._temp_dir
        assert temp_dir.exists()
        assert staged.records == 2
    assert not temp_dir.exists()


def test_confidence_findings_and_empty_labels_are_preserved(tmp_path):
    path = tmp_path / "confidence.csv"
    path.write_text(
        "annotator_id,item_id,label,confidence\n"
        "A01,it_1,topic_alpha,not_a_number\n"
        "A02,it_1,topic_alpha,1.7\n"
        "A03,it_1,,0.5\n",
        encoding="utf-8",
    )
    spec = _spec(path)
    memory = load_dataset(spec)
    with stage_dataset(spec, chunk_size=9) as staged:
        assert _rows(staged) == _rows(memory)
        assert _findings(staged) == _findings(memory)
        codes = [item["code"] for item in _findings(staged)]
        assert codes == ["INVALID_CONFIDENCE", "INVALID_CONFIDENCE"]
        assert staged.missing_annotations == 1


def test_multilabel_jsonl_pipeline_equivalence(tmp_path):
    path = tmp_path / "multi.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"annotator_id": "A01", "item_id": "it_1",
                            "label": ["topic_beta", "topic_alpha"]}),
                json.dumps({"annotator_id": "A02", "item_id": "it_1",
                            "label": ["topic_alpha"]}),
                json.dumps({"annotator_id": "A01", "item_id": "it_2",
                            "label": ["topic_beta"]}),
                json.dumps({"annotator_id": "A02", "item_id": "it_2",
                            "label": ["topic_beta"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = _spec(path, fmt="jsonl")
    config = load_config(_jsonl_config(tmp_path, path))
    memory = load_dataset(spec)
    with stage_dataset(spec, chunk_size=13) as staged:
        assert _rows(staged) == _rows(memory)
        assert validate_dataset(staged, config).to_dict() == validate_dataset(
            memory, config
        ).to_dict()
        assert compute_metric(staged, config).to_dict() == compute_metric(
            memory, config
        ).to_dict()


def _jsonl_config(tmp_path, path):
    config_path = make_config(tmp_path, path, multi_label=True)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["dataset"] = {"path": str(path), "format": "jsonl", "encoding": "utf-8"}
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path
