from __future__ import annotations

import json

from tagconsensus.adjudicate import build_evidence_basis
from tagconsensus.cli import run_demo
from tagconsensus.config import load_config
from tagconsensus.report import dump_json, write_consensus, write_reports
from tagconsensus.models import ConsensusRecord
from tagconsensus.loader import load_dataset

from conftest import make_config, write_jsonl


def _pipeline_config(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_beta\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_beta\n"
        "A01,it_3,topic_alpha\nA02,it_3,topic_beta\nA03,it_3,topic_gamma\n",
        encoding="utf-8",
    )
    pre_config = load_config(make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], min_annotators=2,
        review_threshold=0.8,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    ))
    dataset = load_dataset(pre_config.dataset)
    basis = build_evidence_basis(dataset, pre_config)
    bound = {
        "item_id": "it_3", "decision_label": "topic_gamma",
        "reviewer": "reviewer_01", "reason": "tie resolved",
        "evidence": {
            "binding_schema_version": basis.binding_schema_version,
            "binding_scope": basis.binding_scope,
            "item": {"summary": basis.item_evidence["it_3"],
                     "fingerprint": basis.item_fingerprints["it_3"]},
            "scheme": {"fields": ["label_space", "multi_label",
                                  "min_annotators_per_item", "review_threshold"],
                       "summary": basis.scheme_summary,
                       "fingerprint": basis.scheme_fingerprint},
        },
    }
    decisions = write_jsonl(tmp_path / "adjudications.jsonl", [bound])
    return make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], min_annotators=2,
        review_threshold=0.8, adjudication_file=decisions,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )


def test_report_contains_required_sections(tmp_path):
    config_path = _pipeline_config(tmp_path)
    assert run_demo(str(config_path)) == 0
    output = tmp_path / "output"
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert report["conclusion"]["code"] == "review"
    assert report["data"]["record_count"] == 9
    assert report["validation"]["status"] == "pass"
    assert report["metrics"]["name"] == "pairwise_agreement"
    assert report["metrics"]["Do"] is not None
    assert report["metrics"]["eligible_items"] == 3
    statuses = {item["item_id"]: item["status"] for item in report["review_queue"]}
    assert statuses == {"it_1": "open", "it_3": "resolved"}
    assert report["consensus"]["source_counts"] == {
        "majority": 1, "adjudicated": 1, "unresolved": 1, "adjudication_stale": 0
    }
    assert report["adjudications"]["adopted"] == 1
    assert report["adjudications"]["valid"] == 1
    assert report["adjudications"]["stale"] == 0
    assert report["adjudication_binding"]["scope"] == "per_item"
    assert report["adjudications"]["resolved_review_items"] == 1
    assert len(report["leave_one_out"]) == 3
    markdown = (output / "report.md").read_text(encoding="utf-8")
    for section in ("一致性度量", "复核队列", "标注人", "留一影响度", "裁决与共识输出"):
        assert section in markdown
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "<style>" in html
    assert "<link" not in html
    assert ("http" + "://") not in html
    assert ("https" + "://") not in html
    consensus = [
        json.loads(line)
        for line in (output / "consensus.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["source"] for record in consensus] == [
        "unresolved", "majority", "adjudicated"
    ]


def test_report_json_is_deterministic_except_run_fields(tmp_path):
    config_path = _pipeline_config(tmp_path)
    assert run_demo(str(config_path)) == 0
    first = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    assert run_demo(str(config_path)) == 0
    second = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    for report in (first, second):
        report["run"].pop("run_id")
        report["run"].pop("started_at")
        report["run"].pop("completed_at")
    assert first == second


def test_reports_are_byte_stable_for_identical_report(tmp_path):
    report = {
        "schema_version": "1.0", "tool": "tagconsensus",
        "run": {"run_id": "fixed", "started_at": "t0", "completed_at": "t1",
                "error_code": None},
        "config": {"fingerprint": "f", "output_dir": str(tmp_path),
                   "config_path": str(tmp_path / "config.yaml")},
        "data": {"file_sha256": "d", "record_count": 0,
                 "annotator_fingerprint": "a", "items": 0, "annotators": [],
                 "missing_annotations": 0},
        "validation": {"status": "pass", "error_count": 0, "warning_count": 0,
                       "promoted_count": 0, "findings": []},
        "metrics": None, "review_queue": [], "annotators": [], "leave_one_out": [],
        "adjudication_binding": {
            "scope": None, "binding_schema_version": None,
            "scheme_fields": ["label_space", "multi_label",
                              "min_annotators_per_item", "review_threshold"],
            "scheme_fingerprint": None, "scheme_summary": None,
            "item_fingerprints": {},
        },
        "adjudications": {"provided": 0, "adopted": 0, "valid": 0, "stale": 0,
                          "stale_reason_counts": {},
                          "resolved_review_items": 0, "items": []},
        "consensus": {"source_counts": {"majority": 0, "adjudicated": 0,
                                        "unresolved": 0,
                                        "adjudication_stale": 0}, "records": []},
        "error": None,
        "conclusion": {"code": "pass", "label": "通过", "text": "ok"},
    }
    first = dump_json(report)
    second = dump_json(report)
    assert first == second
    assert first.endswith("\n")
    artifacts = write_reports(report, tmp_path / "reports")
    assert len(artifacts) == 3


def test_consensus_output_is_sorted_and_stable(tmp_path):
    records = [
        ConsensusRecord("it_2", ("topic_beta",), "majority", 1.0, ()),
        ConsensusRecord("it_1", (), "unresolved", 0.5, ("TIE",)),
    ]
    artifact = write_consensus(records, tmp_path)
    text = (tmp_path / "consensus.jsonl").read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert json.loads(lines[0])["item_id"] == "it_1"
    assert json.loads(lines[1])["item_id"] == "it_2"
    assert artifact["size_bytes"] == len(text.encode("utf-8"))
