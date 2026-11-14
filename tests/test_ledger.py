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
