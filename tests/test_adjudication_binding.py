"""End-to-end evidence binding scenarios through the offline CLI and ledger."""

from __future__ import annotations

import json
import sqlite3

from tagconsensus.cli import main
from tagconsensus.ledger import get_run, latest_run

from conftest import make_config, write_jsonl

ROWS = [
    ("A01", "it_1", "topic_alpha"),
    ("A02", "it_1", "topic_alpha"),
    ("A03", "it_1", "topic_beta"),
    ("A01", "it_2", "topic_beta"),
    ("A02", "it_2", "topic_beta"),
    ("A03", "it_2", "topic_beta"),
    ("A01", "it_3", "topic_alpha"),
    ("A02", "it_3", "topic_beta"),
    ("A03", "it_3", "topic_gamma"),
]


def _write_data(tmp_path, rows, name="data.csv"):
    path = tmp_path / name
    lines = ["annotator_id,item_id,label"]
    lines.extend(f"{a},{i},{l}" for a, i, l in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _config(tmp_path, data, adjudication, **overrides):
    settings = {
        "metric": "pairwise_agreement",
        "annotators": ["A01", "A02", "A03"], "min_annotators": 2,
        "review_threshold": 0.8, "adjudication_file": adjudication,
        "output_dir": tmp_path / "output",
        "db_path": tmp_path / "var" / "tagconsensus.db",
    }
    settings.update(overrides)
    return make_config(tmp_path, data, **settings)


def _bind(tmp_path, data, raw_decisions_path, **overrides):
    config_path = _config(tmp_path, data, raw_decisions_path, **overrides)
    bound_path = tmp_path / "bound.jsonl"
    rc = main([
        "bind-adjudication", "--config", str(config_path),
        "--output", str(bound_path),
    ])
    assert rc == 0
    return bound_path


def _read_consensus(output_dir):
    return {
        row["item_id"]: row
        for row in (
            json.loads(line)
            for line in (output_dir / "consensus.jsonl")
            .read_text(encoding="utf-8").splitlines()
        )
    }


def _read_report(output_dir):
    return json.loads((output_dir / "report.json").read_text(encoding="utf-8"))


def _decisions():
    return [
        {"item_id": "it_1", "decision_label": "topic_gamma",
         "reviewer": "reviewer_01", "reason": "low consensus override"},
        {"item_id": "it_3", "decision_label": "topic_gamma",
         "reviewer": "reviewer_02", "reason": "tie resolved"},
    ]


def test_bound_decisions_are_adopted_and_reported_in_all_formats(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(tmp_path, data, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "adjudications=2 valid=2 stale=0" in out
    consensus = _read_consensus(tmp_path / "output")
    assert consensus["it_1"]["source"] == "adjudicated"
    assert consensus["it_1"]["labels"] == ["topic_gamma"]
    report = _read_report(tmp_path / "output")
    assert report["adjudication_binding"]["scope"] == "per_item"
    assert report["adjudications"]["valid"] == 2
    assert report["adjudications"]["stale"] == 0
    markdown = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
    assert "裁决依据绑定" in markdown and "per_item" in markdown
    html = (tmp_path / "output" / "report.html").read_text(encoding="utf-8")
    assert "裁决依据绑定" in html and "失效待重新核验 0 条" in html
    db = sqlite3.connect(tmp_path / "var" / "tagconsensus.db")
    try:
        rows = db.execute(
            "SELECT item_id, status, stale_reasons, item_evidence_fingerprint, "
            "scheme_fingerprint FROM adjudications ORDER BY item_id"
        ).fetchall()
    finally:
        db.close()
    assert [row[0] for row in rows] == ["it_1", "it_3"]
    assert all(row[1] == "valid" for row in rows)
    assert all(row[3] and row[4] for row in rows)


def test_revised_item_votes_invalidate_only_that_decision(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(tmp_path, data, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    capsys.readouterr()
    # Revise it_1: A03 flips beta -> alpha (unanimous alpha now). it_3 untouched.
    revised_rows = [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_alpha"),
        *ROWS[3:],
    ]
    revised = _write_data(tmp_path, revised_rows, name="data_v2.csv")
    config_path = _config(tmp_path, revised, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "adjudications=2 valid=1 stale=1" in out
    assert "stale_adjudication item=it_1" in out
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in out
    consensus = _read_consensus(tmp_path / "output")
    stale = consensus["it_1"]
    assert stale["source"] == "adjudication_stale"
    assert stale["labels"] == []
    assert stale["adjudication_status"] == "stale"
    assert stale["prior_decision_label"] == "topic_gamma"
    assert stale["prior_reviewer"] == "reviewer_01"
    assert stale["prior_reason"] == "low consensus override"
    assert stale["stale_reasons"] == ["ADJ_ITEM_EVIDENCE_CHANGED"]
    # The unrelated decision stays adopted; no silent majority downgrade.
    assert consensus["it_3"]["source"] == "adjudicated"
    assert consensus["it_3"]["labels"] == ["topic_gamma"]
    report = _read_report(tmp_path / "output")
    assert report["adjudications"]["stale"] == 1
    assert report["adjudications"]["stale_reason_counts"] == {
        "ADJ_ITEM_EVIDENCE_CHANGED": 1
    }
    queue = {item["item_id"]: item for item in report["review_queue"]}
    assert queue["it_1"]["status"] == "stale"
    assert queue["it_1"]["prior_decision_label"] == "topic_gamma"
    markdown = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in markdown
    assert "失效 1 条" in markdown
    html = (tmp_path / "output" / "report.html").read_text(encoding="utf-8")
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in html
    # Conclusion must force re-review.
    assert report["conclusion"]["code"] == "review"
    db = sqlite3.connect(tmp_path / "var" / "tagconsensus.db")
    try:
        row = db.execute(
            "SELECT status, stale_reasons FROM adjudications "
            "WHERE item_id = 'it_1' ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        run_row = db.execute(
            "SELECT adjudications_valid, adjudications_stale, binding_scope "
            "FROM runs ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    finally:
        db.close()
    assert row[0] == "stale"
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in row[1]
    assert run_row == (1, 1, "per_item")


def test_unrelated_item_revision_keeps_decisions_valid(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    # Only it_2 changes; neither bound item loses validity.
    revised_rows = [
        *ROWS[:3],
        ("A01", "it_2", "topic_alpha"),
        ("A02", "it_2", "topic_beta"),
        ("A03", "it_2", "topic_beta"),
        *ROWS[6:],
    ]
    revised = _write_data(tmp_path, revised_rows, name="data_v2.csv")
    config_path = _config(tmp_path, revised, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "adjudications=2 valid=2 stale=0" in out
    consensus = _read_consensus(tmp_path / "output")
    assert consensus["it_1"]["source"] == "adjudicated"
    assert consensus["it_3"]["source"] == "adjudicated"


def test_review_threshold_change_invalidates_every_decision(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(tmp_path, data, bound, review_threshold=0.5)
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "adjudications=2 valid=0 stale=2" in out
    assert out.count("ADJ_SCHEME_CHANGED") == 2
    report = _read_report(tmp_path / "output")
    assert report["adjudications"]["stale_reason_counts"] == {
        "ADJ_SCHEME_CHANGED": 2
    }
    consensus = _read_consensus(tmp_path / "output")
    for item_id in ("it_1", "it_3"):
        assert consensus[item_id]["source"] == "adjudication_stale"
        assert consensus[item_id]["labels"] == []
        assert consensus[item_id]["prior_decision_label"] is not None


def test_label_space_change_invalidates_decisions(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(
        tmp_path, data, bound,
        label_space=["topic_alpha", "topic_beta", "topic_gamma",
                     "topic_delta", "other", "topic_epsilon"],
    )
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "valid=0 stale=2" in out
    assert "ADJ_SCHEME_CHANGED" in out


def test_legacy_unbound_file_is_stale_not_applied(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    legacy = write_jsonl(tmp_path / "legacy.jsonl", _decisions())
    config_path = _config(tmp_path, data, legacy)
    assert main(["demo", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "valid=0 stale=2" in out
    assert "ADJ_EVIDENCE_MISSING" in out
    consensus = _read_consensus(tmp_path / "output")
    assert consensus["it_1"]["source"] == "adjudication_stale"
    assert consensus["it_1"]["prior_reason"] == "low consensus override"


def test_history_readback_keeps_binding_state_across_runs(tmp_path, capsys):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(tmp_path, data, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    revised_rows = [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_alpha"),
        *ROWS[3:],
    ]
    revised = _write_data(tmp_path, revised_rows, name="data_v2.csv")
    assert main(["demo", "--config", str(_config(tmp_path, revised, bound))]) == 0
    capsys.readouterr()
    db_path = tmp_path / "var" / "tagconsensus.db"
    detail = get_run(db_path, latest_run(db_path)["run_id"])
    assert detail is not None
    assert detail["run"]["adjudications_valid"] == 1
    assert detail["run"]["adjudications_stale"] == 1
    assert detail["run"]["binding_scope"] == "per_item"
    assert len(detail["run"]["decision_scheme_fingerprint"]) == 64
    by_item = {row["item_id"]: row for row in detail["adjudications"]}
    assert by_item["it_1"]["status"] == "stale"
    assert "ADJ_ITEM_EVIDENCE_CHANGED" in by_item["it_1"]["stale_reasons"]
    assert by_item["it_3"]["status"] == "valid"
    assert bool(by_item["it_3"]["adopted"]) is True
    assert bool(by_item["it_1"]["adopted"]) is False
    # Reviewer, reason and old decision remain traceable in the ledger.
    assert by_item["it_1"]["reviewer"] == "reviewer_01"
    assert by_item["it_1"]["reason"] == "low consensus override"
    assert by_item["it_1"]["decision_label"] == "topic_gamma"
    assert main(["history", "--db", str(db_path)]) == 0
    assert "run_id" in capsys.readouterr().out


def test_rerun_without_changes_is_byte_stable_except_run_fields(tmp_path):
    data = _write_data(tmp_path, ROWS)
    raw = write_jsonl(tmp_path / "raw_decisions.jsonl", _decisions())
    bound = _bind(tmp_path, data, raw)
    config_path = _config(tmp_path, data, bound)
    assert main(["demo", "--config", str(config_path)]) == 0
    first = _read_report(tmp_path / "output")
    assert main(["demo", "--config", str(config_path)]) == 0
    second = _read_report(tmp_path / "output")
    for report in (first, second):
        report["run"].pop("run_id")
        report["run"].pop("started_at")
        report["run"].pop("completed_at")
    assert first == second
    consensus_first = (tmp_path / "output" / "consensus.jsonl").read_text(
        encoding="utf-8"
    )
    assert main(["demo", "--config", str(config_path)]) == 0
    assert (tmp_path / "output" / "consensus.jsonl").read_text(
        encoding="utf-8"
    ) == consensus_first
