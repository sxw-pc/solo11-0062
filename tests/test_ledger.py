from __future__ import annotations

import sqlite3

from tagconsensus.cli import run_demo
from tagconsensus.ledger import get_run, latest_run, list_runs

from conftest import make_config


def _write_data(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_alpha\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_beta\n",
        encoding="utf-8",
    )
    return path


def _config(tmp_path):
    return make_config(
        tmp_path, _write_data(tmp_path), metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"],
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )


def test_empty_ledger_is_not_an_error(tmp_path):
    missing = tmp_path / "none.db"
    assert list_runs(missing) == []
    assert latest_run(missing) is None
    assert get_run(missing, "abc") is None


def test_demo_records_all_tables(tmp_path):
    config_path = _config(tmp_path)
    assert run_demo(str(config_path)) == 0
    db_path = tmp_path / "var" / "tagconsensus.db"
    runs = list_runs(db_path)
    assert len(runs) == 1
    detail = get_run(db_path, runs[0]["run_id"])
    assert detail is not None
    assert detail["run"]["conclusion"] == "pass"
    assert detail["run"]["metric_name"] == "pairwise_agreement"
    assert len(detail["metrics"]) == 1
    assert len(detail["annotators"]) == 3
    assert len(detail["artifacts"]) == 4
    assert detail["review_queue"] == []
    assert detail["adjudications"] == []
    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()
    assert {
        "runs", "validation_findings", "metrics", "review_queue",
        "annotators", "adjudications", "artifacts",
    } <= tables


def test_artifact_metadata_matches_files(tmp_path):
    config_path = _config(tmp_path)
    run_demo(str(config_path))
    db_path = tmp_path / "var" / "tagconsensus.db"
    detail = get_run(db_path, latest_run(db_path)["run_id"])
    assert {item["kind"] for item in detail["artifacts"]} == {
        "report_json", "report_md", "report_html", "consensus"
    }
    for artifact in detail["artifacts"]:
        from pathlib import Path

        path = Path(artifact["path"])
        assert path.exists()
        assert path.stat().st_size == artifact["size_bytes"]
        assert len(artifact["sha256"]) == 64


def test_two_runs_are_listed(tmp_path):
    config_path = _config(tmp_path)
    assert run_demo(str(config_path)) == 0
    assert run_demo(str(config_path)) == 0
    db_path = tmp_path / "var" / "tagconsensus.db"
    runs = list_runs(db_path)
    assert len(runs) == 2
    assert {run["metric_name"] for run in runs} == {"pairwise_agreement"}
    assert latest_run(db_path)["run_id"] in {run["run_id"] for run in runs}


def test_legacy_database_migrates_and_keeps_history_readable(tmp_path):
    import sqlite3

    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE runs (
          run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
          config_path TEXT, config_fingerprint TEXT, data_sha256 TEXT,
          data_records INTEGER, annotator_fingerprint TEXT, metric_name TEXT,
          metric_value REAL, conclusion TEXT, conclusion_label TEXT,
          validation_status TEXT, error_code TEXT, report_json_path TEXT,
          report_md_path TEXT, report_html_path TEXT, consensus_path TEXT,
          output_dir TEXT
        );
        CREATE TABLE adjudications (
          run_id TEXT NOT NULL, item_id TEXT NOT NULL, decision_label TEXT,
          reviewer TEXT, reason TEXT, adopted INTEGER,
          PRIMARY KEY (run_id, item_id)
        );
        INSERT INTO runs (run_id, started_at)
        VALUES ('legacy-run', '2026-01-01T00:00:00+00:00');
        INSERT INTO adjudications
        (run_id, item_id, decision_label, reviewer, reason, adopted)
        VALUES ('legacy-run', 'it_1', 'topic_alpha', 'old_reviewer', 'old', 1);
        """
    )
    connection.commit()
    connection.close()

    # A fresh write (record_run always calls init_db first) migrates the
    # legacy schema while preserving the historical rows.
    from tagconsensus.ledger import init_db

    init_db(db_path)
    runs = list_runs(db_path)
    assert len(runs) == 1
    assert runs[0]["run_id"] == "legacy-run"
    assert runs[0]["binding_scope"] is None
    assert runs[0]["adjudications_valid"] == 0
    detail = get_run(db_path, "legacy-run")
    legacy = detail["adjudications"][0]
    assert legacy["status"] is None
    assert legacy["stale_reasons"] is None
    assert legacy["item_evidence_fingerprint"] is None
    assert legacy["decision_label"] == "topic_alpha"
    assert legacy["reviewer"] == "old_reviewer"
