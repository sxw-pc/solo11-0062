from __future__ import annotations

import json
import sqlite3

from tagconsensus.cli import run_demo
from tagconsensus.config import load_config
from tagconsensus.items import build_items
from tagconsensus.ledger import get_run, latest_run
from tagconsensus.metrics import compute_metric
from tagconsensus.validate import validate_dataset
from tagconsensus.webapp import create_app

from conftest import make_config, make_dataset, write_manifest_csv


def _rows():
    return [
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A03", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
    ]


def _data(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_beta\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\n",
        encoding="utf-8",
    )
    return path


# ---------- strict --------------------------------------------------------


def test_missing_assignment_warning_blocks_in_strict(tmp_path):
    data = _data(tmp_path)
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
        ("A01", "it_2"), ("A02", "it_2"), ("A03", "it_2"),
    ])
    config = load_config(make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], strict=True,
        assignment_manifest=manifest,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    ))
    dataset = make_dataset(_rows())
    result = validate_dataset(dataset, config)
    assert result.blocked is True
    promoted = [
        item for item in result.findings
        if item.code == "ASSIGNMENT_MISSING" and item.promoted
    ]
    assert {(item.annotator_id, item.item_id) for item in promoted} == {
        ("A03", "it_2")
    }
    assert all(item.severity == "error" for item in promoted)


def test_missing_assignment_does_not_block_without_strict(tmp_path):
    data = _data(tmp_path)
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
        ("A01", "it_2"), ("A02", "it_2"), ("A03", "it_2"),
    ])
    config = load_config(make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], strict=False,
        assignment_manifest=manifest,
    ))
    result = validate_dataset(make_dataset(_rows()), config)
    missing = [item for item in result.findings if item.code == "ASSIGNMENT_MISSING"]
    assert missing and missing[0].severity == "warning"
    assert result.blocked is False
    assert result.status == "warn"


# ---------- statistics stay unchanged -------------------------------------


def test_metrics_and_consensus_are_invariant_across_modes(tmp_path):
    data = _data(tmp_path)
    base_config = load_config(make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"],
    ))
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
        ("A01", "it_2"), ("A02", "it_2"), ("A03", "it_2"),
    ])
    manifest_config = load_config(make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], assignment_manifest=manifest,
    ))
    dataset = make_dataset(_rows())
    base_metric = compute_metric(dataset, base_config)
    manifest_metric = compute_metric(dataset, manifest_config)
    assert base_metric.value == manifest_metric.value
    assert base_metric.eligible_annotations == manifest_metric.eligible_annotations
    base_items = build_items(dataset, base_config)
    manifest_items = build_items(dataset, manifest_config)
    assert [item.to_dict() for item in base_items] == [
        item.to_dict() for item in manifest_items
    ]
    # But coverage audit differs: manifest mode reports the missing pair.
    base_validation = validate_dataset(dataset, base_config)
    manifest_validation = validate_dataset(dataset, manifest_config)
    assert base_validation.assignment_mode == "full"
    assert manifest_validation.assignment_mode == "manifest"
    assert not [
        item for item in base_validation.findings
        if item.code == "UNASSIGNED_ANNOTATION"
    ]
    assert manifest_validation.coverage["missing_pairs"] == 1
    assert base_validation.coverage["missing_pairs"] == 1


# ---------- end-to-end report + ledger ------------------------------------


def test_pipeline_persists_assignment_coverage(tmp_path):
    data = _data(tmp_path)
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
        ("A01", "it_2"), ("A02", "it_2"), ("A03", "it_2"),
    ])
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], assignment_manifest=manifest,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )
    assert run_demo(str(config_path)) == 0
    report = json.loads(
        (tmp_path / "output" / "report.json").read_text(encoding="utf-8")
    )
    coverage = report["assignment_coverage"]
    assert coverage["mode"] == "manifest"
    assert coverage["expected_pairs"] == 6
    assert coverage["missing_pairs"] == 1
    assert coverage["submitted_pairs"] == 5
    assert coverage["manifest_path"].endswith("assign.csv")
    codes = {item["code"] for item in coverage["findings"]}
    assert "ASSIGNMENT_MISSING" in codes
    missing = next(
        item for item in coverage["findings"]
        if item["code"] == "ASSIGNMENT_MISSING"
    )
    assert missing["item_id"] == "it_2" and missing["annotator_id"] == "A03"
    # Findings also live in the standard validation section.
    all_codes = {item["code"] for item in report["validation"]["findings"]}
    assert "ASSIGNMENT_MISSING" in all_codes
    markdown = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
    assert "分配覆盖审计" in markdown and "ASSIGNMENT_MISSING" in markdown
    html = (tmp_path / "output" / "report.html").read_text(encoding="utf-8")
    assert "分配覆盖审计" in html and "ASSIGNMENT_MISSING" in html

    db_path = tmp_path / "var" / "tagconsensus.db"
    detail = get_run(db_path, latest_run(db_path)["run_id"])
    assert detail["assignment_coverage"][0]["mode"] == "manifest"
    assert detail["assignment_coverage"][0]["missing_pairs"] == 1
    stored = {
        (item["item_id"], item["annotator_id"], item["code"])
        for item in detail["findings"]
    }
    assert ("it_2", "A03", "ASSIGNMENT_MISSING") in stored
    connection = sqlite3.connect(db_path)
    try:
        tables = {
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
    finally:
        connection.close()
    assert "assignment_coverage" in tables


def test_unassigned_record_blocks_pipeline_and_reports_identity(tmp_path):
    data = _data(tmp_path)
    # Manifest only covers it_1; every it_2 row is unassigned.
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
    ])
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], assignment_manifest=manifest,
        output_dir=tmp_path / "output_blocked",
        db_path=tmp_path / "var" / "blocked.db",
    )
    assert run_demo(str(config_path)) == 2
    report = json.loads(
        (tmp_path / "output_blocked" / "report.json").read_text(encoding="utf-8")
    )
    assert report["conclusion"]["code"] == "blocked"
    codes = {item["code"] for item in report["validation"]["findings"]}
    assert "UNASSIGNED_ANNOTATION" in codes
    unassigned = [
        item for item in report["validation"]["findings"]
        if item["code"] == "UNASSIGNED_ANNOTATION"
    ]
    assert {item["annotator_id"] for item in unassigned} == {"A01", "A02"}
    assert all(item["item_id"] == "it_2" for item in unassigned)


# ---------- web surfaces --------------------------------------------------


def test_review_page_renders_assignment_section(tmp_path):
    data = _data(tmp_path)
    manifest = write_manifest_csv(tmp_path / "assign.csv", [
        ("A01", "it_1"), ("A02", "it_1"), ("A03", "it_1"),
        ("A01", "it_2"), ("A02", "it_2"), ("A03", "it_2"),
    ])
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], assignment_manifest=manifest,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )
    run_demo(str(config_path))
    db_path = tmp_path / "var" / "tagconsensus.db"
    app = create_app(str(db_path))
    app.config.update(TESTING=True)
    client = app.test_client()
    review_body = client.get("/review").get_data(as_text=True)
    assert "分配覆盖审计" in review_body
    assert "ASSIGNMENT_MISSING" in review_body
    assert "A03" in review_body and "it_2" in review_body
    ledger_body = client.get(
        f"/ledger?run_id={latest_run(db_path)['run_id']}"
    ).get_data(as_text=True)
    assert "分配覆盖审计" in ledger_body
    # The ledger findings table now carries annotator identity.
    assert "<th>annotator</th>" in ledger_body


def test_discovery_run_has_no_assignment_section(tmp_path):
    config_path = make_config(
        tmp_path, _data(tmp_path), metric="pairwise_agreement",
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )
    run_demo(str(config_path))
    report = json.loads(
        (tmp_path / "output" / "report.json").read_text(encoding="utf-8")
    )
    assert report["assignment_coverage"] is None
    app = create_app(str(tmp_path / "var" / "tagconsensus.db"))
    app.config.update(TESTING=True)
    client = app.test_client()
    assert "分配覆盖审计" not in client.get("/review").get_data(as_text=True)
