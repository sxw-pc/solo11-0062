from __future__ import annotations

import json

from tagconsensus.cli import main

from conftest import make_config


def _write_clean(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_alpha\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_beta\n",
        encoding="utf-8",
    )
    return path


def _clean_config(tmp_path):
    return make_config(
        tmp_path, _write_clean(tmp_path), metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"],
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )


def test_demo_command_exit_zero_and_writes_outputs(tmp_path, capsys):
    config_path = _clean_config(tmp_path)
    assert main(["demo", "--config", str(config_path)]) == 0
    output = tmp_path / "output"
    assert {
        "report.json", "report.md", "report.html", "consensus.jsonl"
    } <= {path.name for path in output.iterdir()}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["conclusion"]["code"] == "pass"
    captured = capsys.readouterr()
    assert "conclusion=通过" in captured.out
    assert "run_id=" in captured.out


def test_strict_config_blocks_warnings(tmp_path, capsys):
    data = tmp_path / "warn.csv"
    data.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,\nA03,it_1,topic_alpha\n",
        encoding="utf-8",
    )
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement", strict=True,
        annotators=["A01", "A02", "A03"],
        output_dir=tmp_path / "strict-output",
        db_path=tmp_path / "var" / "strict.db",
    )
    assert main(["demo", "--config", str(config_path)]) == 2
    report = json.loads(
        (tmp_path / "strict-output" / "report.json").read_text(encoding="utf-8")
    )
    assert report["conclusion"]["code"] == "blocked"
    assert report["conclusion"]["label"] == "阻断"
    assert report["validation"]["error_count"] >= 1
    assert report["validation"]["promoted_count"] >= 1
    assert "conclusion=阻断" in capsys.readouterr().out


def test_history_lists_two_runs(tmp_path, capsys):
    config_path = _clean_config(tmp_path)
    assert main(["demo", "--config", str(config_path)]) == 0
    assert main(["demo", "--config", str(config_path)]) == 0
    capsys.readouterr()
    db_path = tmp_path / "var" / "tagconsensus.db"
    assert main(["history", "--db", str(db_path)]) == 0
    output = capsys.readouterr().out
    assert "run_id\tstarted_at\tmetric\tvalue\tconclusion\trecords" in output
    assert output.count("pairwise_agreement") == 2


def test_history_without_db_is_empty(tmp_path, capsys):
    assert main(["history", "--db", str(tmp_path / "none.db")]) == 0
    assert "no runs" in capsys.readouterr().out


def test_config_error_exit_code(tmp_path, capsys):
    assert main(["demo", "--config", str(tmp_path / "missing.yaml")]) == 2
    assert "CFG_FILE_NOT_FOUND" in capsys.readouterr().err
