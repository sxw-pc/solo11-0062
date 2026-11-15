from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tagconsensus.cli import main, run_demo
from tagconsensus.config import load_config
from tagconsensus.finding_history import compare_findings
from tagconsensus.ledger import get_run, init_db, latest_run
from tagconsensus.loader import load_dataset
from tagconsensus.validate import validate_dataset

from conftest import make_config, make_dataset

HEADER = "annotator_id,item_id,label\n"


def _run(tmp_path, rows: str, *, persist: bool = False, **overrides) -> str:
    data = tmp_path / "data.csv"
    data.write_text(HEADER + rows, encoding="utf-8")
    extra = {"persist_raw_rows": True} if persist else None
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
        extra=extra, **overrides,
    )
    run_demo(str(config_path))
    return str(tmp_path / "var" / "tagconsensus.db")


def _compare(db_path) -> dict:
    return compare_findings(Path(db_path))


def test_duplicate_key_produces_distinct_stable_findings(tmp_path):
    db = _run(
        tmp_path,
        "A01,it_1,topic_alpha\nA01,it_1,topic_alpha\n"
        "A01,it_1,topic_beta\nA02,it_1,topic_alpha\n",
    )
    run = latest_run(Path(db))
    detail = get_run(Path(db), run["run_id"])
    duplicates = [
        item for item in detail["findings"] if item["code"] == "DUPLICATE_ANNOTATION"
    ]
    assert len(duplicates) == 2
    keys = {item["finding_key"] for item in duplicates}
    assert len(keys) == 2 and None not in keys
    # Removing one duplicate resolves exactly one finding; the other persists.
    _run(
        tmp_path,
        "A01,it_1,topic_alpha\nA01,it_1,topic_beta\nA02,it_1,topic_alpha\n",
    )
    result = _compare(db)
    assert result["comparable"] is True
    assert result["counts"]["new"] == 0
    assert result["counts"]["persisting"] == 1
    assert result["counts"]["resolved"] == 1
    assert result["resolved"][0]["code"] == "DUPLICATE_ANNOTATION"


def test_line_reorder_keeps_identity_and_flags_only_line_change(tmp_path):
    _run(
        tmp_path,
        "A01,it_1,\nA02,it_1,topic_alpha\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\n",
    )
    _run(
        tmp_path,
        "A02,it_2,topic_beta\nA01,it_1,\n"
        "A02,it_1,topic_alpha\nA01,it_2,topic_beta\n",
    )
    result = _compare(tmp_path / "var" / "tagconsensus.db")
    assert result["comparable"] is True
    counts = result["counts"]
    assert counts["new"] == 0 and counts["resolved"] == 0
    assert counts["persisting"] == 2  # EMPTY_LABEL + INSUFFICIENT_ANNOTATORS
    assert counts["content_changed"] == 0
    moved = [item for item in result["persisting"] if item["line_changed"]]
    assert len(moved) == 1
    assert moved[0]["code"] == "EMPTY_LABEL"
    assert moved[0]["previous_line_no"] == 2
    assert moved[0]["current_line_no"] == 3
    anchored = next(
        item for item in result["persisting"] if item["code"] == "INSUFFICIENT_ANNOTATORS"
    )
    assert anchored["line_changed"] is False


def test_label_revision_is_content_change_not_resolved_plus_new(tmp_path):
    _run(tmp_path, "A01,it_1,not_a_label\nA02,it_1,topic_alpha\n")
    _run(tmp_path, "A01,it_1,still_bad\nA02,it_1,topic_alpha\n")
    result = _compare(tmp_path / "var" / "tagconsensus.db")
    assert result["comparable"] is True
    counts = result["counts"]
    assert counts["new"] == 0 and counts["resolved"] == 0
    assert counts["persisting"] == 1
    entry = result["persisting"][0]
    assert entry["code"] == "LABEL_OUT_OF_SPACE"
    assert entry["content_changed"] is True
    assert entry["row_changed"] is True
    assert entry["line_changed"] is False
    # Actually fixing the label resolves the finding.
    _run(tmp_path, "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\n")
    fixed = _compare(tmp_path / "var" / "tagconsensus.db")
    assert fixed["counts"]["resolved"] == 1
    assert fixed["resolved"][0]["code"] == "LABEL_OUT_OF_SPACE"
    assert fixed["counts"]["new"] == 0


def test_config_change_is_reported_incompatible(tmp_path):
    _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n", min_annotators=2)
    _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n", min_annotators=3)
    result = _compare(tmp_path / "var" / "tagconsensus.db")
    assert result["comparable"] is False
    assert result["reason"] == "config_incompatible"


def test_first_run_has_no_previous(tmp_path):
    db = _run(tmp_path, "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\n")
    result = _compare(db)
    assert result["comparable"] is False
    assert result["reason"] == "no_previous_run"


OLD_SCHEMA = """
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
  config_path TEXT, config_fingerprint TEXT, data_sha256 TEXT, data_records INTEGER,
  annotator_fingerprint TEXT, metric_name TEXT, metric_value REAL,
  conclusion TEXT, conclusion_label TEXT, validation_status TEXT, error_code TEXT,
  report_json_path TEXT, report_md_path TEXT, report_html_path TEXT,
  consensus_path TEXT, output_dir TEXT
);
CREATE TABLE validation_findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, code TEXT NOT NULL,
  severity TEXT NOT NULL, message TEXT NOT NULL, item_id TEXT, annotator_id TEXT,
  line_no INTEGER, promoted INTEGER NOT NULL DEFAULT 0
);
"""


def _write_old_run(db_path):
    connection = sqlite3.connect(db_path)
    try:
        connection.executescript(OLD_SCHEMA)
        connection.execute(
            "INSERT INTO runs VALUES (" + ",".join("?" * 19) + ")",
            ("oldrun", "2020-01-01T00:00:00+00:00", None, "c", "fp", "sha", 2,
             "afp", None, None, "review", "需复核", "warn", None,
             None, None, None, None, "out"),
        )
        connection.execute(
            "INSERT INTO validation_findings "
            "(run_id,code,severity,message,item_id,annotator_id,line_no,promoted) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("oldrun", "EMPTY_LABEL", "warning", "legacy", "it_1", "A01", 2, 0),
        )
        connection.commit()
    finally:
        connection.close()


def test_legacy_run_is_migrated_and_reported_incomparable(tmp_path):
    db_path = tmp_path / "var" / "tagconsensus.db"
    db_path.parent.mkdir(parents=True)
    _write_old_run(db_path)
    init_db(db_path)  # migrates the old schema in place
    connection = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(validation_findings)")
        }
        legacy = connection.execute(
            "SELECT finding_key FROM validation_findings WHERE run_id = 'oldrun'"
        ).fetchone()
    finally:
        connection.close()
    assert {"finding_key", "content_key", "row_digest", "row_json"} <= columns
    assert legacy[0] is None  # old row keeps NULL identity material
    db = _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n")
    result = _compare(db)
    assert result["comparable"] is False
    assert result["reason"] == "missing_identity_material"
    assert "previous run" in result["detail"]
    current = compare_findings(
        Path(db), run_id="oldrun",
        against_run_id=result["run_id"],
    )
    assert current["reason"] == "missing_identity_material"
    assert "current run" in current["detail"]


def test_persist_raw_rows_stores_normalized_row(tmp_path):
    db = _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n", persist=True)
    detail = get_run(
        Path(db), latest_run(Path(db))["run_id"]
    )
    empty = next(item for item in detail["findings"] if item["code"] == "EMPTY_LABEL")
    assert empty["row_digest"] is not None
    assert json.loads(empty["row_json"]) == {
        "annotator_id": "A01", "item_id": "it_1", "labels": [], "confidence": None,
    }
    item_level = next(
        item for item in detail["findings"] if item["code"] == "INSUFFICIENT_ANNOTATORS"
    )
    assert item_level["row_json"] is None  # not locatable to a single record


def test_digest_only_storage_omits_row_payload(tmp_path):
    db = _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n")
    detail = get_run(
        Path(db), latest_run(Path(db))["run_id"]
    )
    empty = next(item for item in detail["findings"] if item["code"] == "EMPTY_LABEL")
    assert empty["finding_key"] and empty["row_digest"]
    assert empty["row_json"] is None


def test_validate_and_ledger_share_finding_identity(tmp_path):
    db = _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n")
    report = json.loads(
        (tmp_path / "output" / "report.json").read_text(encoding="utf-8")
    )
    report_keys = {
        item["finding_key"] for item in report["validation"]["findings"]
    }
    assert None not in report_keys and report_keys
    detail = get_run(
        Path(db), latest_run(Path(db))["run_id"]
    )
    ledger_keys = {item["finding_key"] for item in detail["findings"]}
    assert report_keys == ledger_keys


def test_identity_is_stable_across_reordered_dataset(tmp_path):
    config = load_config(make_config(tmp_path, min_annotators=2))
    first = validate_dataset(
        make_dataset([("A01", "it_1", ""), ("A02", "it_1", "topic_alpha")]), config
    )
    second = validate_dataset(
        make_dataset([("A02", "it_1", "topic_alpha"), ("A01", "it_1", "")]), config
    )
    first_keys = {item.code: item.finding_key for item in first.findings}
    second_keys = {item.code: item.finding_key for item in second.findings}
    assert first_keys == second_keys
    assert first_keys["EMPTY_LABEL"] is not None


def test_invalid_confidence_is_record_locatable(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label,confidence\n"
        "A01,it_1,topic_alpha,1.5\nA02,it_1,topic_alpha,0.5\n",
        encoding="utf-8",
    )
    config = load_config(make_config(tmp_path, path))
    dataset = load_dataset(config.dataset)
    result = validate_dataset(dataset, config)
    finding = next(item for item in result.findings if item.code == "INVALID_CONFIDENCE")
    assert (finding.annotator_id, finding.item_id) == ("A01", "it_1")
    assert finding.finding_key and finding.row_digest


def test_cli_findings_outputs_status_lines(tmp_path, capsys):
    _run(tmp_path, "A01,it_1,bad_label\nA02,it_1,topic_alpha\n")
    _run(tmp_path, "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\n")
    db = str(tmp_path / "var" / "tagconsensus.db")
    capsys.readouterr()
    assert main(["findings", "--db", db]) == 0
    output = capsys.readouterr().out
    assert "comparable=true" in output
    assert "resolved=1" in output
    assert "RESOLVED LABEL_OUT_OF_SPACE item=it_1 annotator=A01 line=2" in output


def test_cli_findings_reports_incomparable(tmp_path, capsys):
    _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n", min_annotators=2)
    _run(tmp_path, "A01,it_1,\nA02,it_1,topic_alpha\n", min_annotators=3)
    db = str(tmp_path / "var" / "tagconsensus.db")
    capsys.readouterr()
    assert main(["findings", "--db", db]) == 0
    assert "comparable=false reason=config_incompatible" in capsys.readouterr().out


def test_cli_findings_without_runs(tmp_path, capsys):
    assert main(["findings", "--db", str(tmp_path / "none.db")]) == 0
    assert "comparable=false reason=no_runs" in capsys.readouterr().out
