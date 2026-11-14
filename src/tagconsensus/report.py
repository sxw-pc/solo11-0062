"""Deterministic JSON, Markdown and self-contained HTML reports."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .hashing import sha256_file
from .report_html import render_html
from .models import (
    AnnotatorStat,
    ConsensusRecord,
    Dataset,
    ItemConsensus,
    LeaveOneOut,
    MetricResult,
)
from .validate import ValidationResult


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: _round(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round(item) for item in value]
    return value


def _review_records(
    items: Iterable[ItemConsensus], consensus: Iterable[ConsensusRecord]
) -> list[dict]:
    sources = {record.item_id: record.source for record in consensus}
    records = []
    for item in sorted(items, key=lambda entry: entry.item_id):
        if item.needs_review:
            record = item.to_dict()
            record["reason"] = item.primary_reason
            record["status"] = (
                "resolved" if sources.get(item.item_id) == "adjudicated" else "open"
            )
            records.append(record)
    return records


def _conclusion(
    validation: ValidationResult, review: list[dict], error: dict | None
) -> dict[str, str]:
    if validation.blocked or error is not None:
        return {"code": "blocked", "label": "阻断",
                "text": "存在阻断级问题，本次结果不得作为共识标签使用。"}
    if validation.warning_count or any(item["status"] == "open" for item in review):
        return {"code": "review", "label": "需复核",
                "text": "数据可用但存在告警或争议条目，必须经人工复核后方可使用。"}
    return {"code": "pass", "label": "通过",
            "text": "校验、度量与争议检查均通过，可按方案使用本批共识标签。"}


def build_report(
    *, config: Config, dataset: Dataset, validation: ValidationResult,
    run_id: str, started_at: str, completed_at: str,
    metric: MetricResult | None = None, items: Iterable[ItemConsensus] = (),
    annotators: Iterable[AnnotatorStat] = (), leave_one_out: Iterable[LeaveOneOut] = (),
    adjudications: list[dict] | None = None,
    consensus: Iterable[ConsensusRecord] = (), error: dict | None = None,
) -> dict[str, Any]:
    consensus_list = [record.to_dict() for record in consensus]
    review = _review_records(items, consensus) if consensus_list else []
    decision_items = adjudications or []
    resolved = sum(1 for item in review if item["status"] == "resolved")
    source_counts = {"majority": 0, "adjudicated": 0, "unresolved": 0}
    for record in consensus_list:
        source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
    result = {
        "schema_version": "1.0", "tool": "tagconsensus",
        "run": {"run_id": run_id, "started_at": started_at,
                "completed_at": completed_at,
                "error_code": error["code"] if error else None},
        "config": config.summary(),
        "data": {**dataset.fingerprint_dict(), "items": len(dataset.item_ids),
                 "annotators": list(dataset.annotators),
                 "missing_annotations": dataset.missing_annotations},
        "validation": validation.to_dict(),
        "metrics": metric.to_dict() if metric is not None else None,
        "review_queue": review,
        "annotators": [
            entry.to_dict()
            for entry in sorted(annotators, key=lambda entry: entry.annotator_id)
        ],
        "leave_one_out": [
            entry.to_dict() for entry in sorted(
                leave_one_out,
                key=lambda entry: (entry.delta is None,
                                   -abs(entry.delta) if entry.delta is not None else 0.0,
                                   entry.annotator_id),
            )
        ],
        "adjudications": {
            "provided": len(decision_items), "adopted": len(decision_items),
            "resolved_review_items": resolved,
            "items": sorted(decision_items, key=lambda entry: entry["item_id"]),
        },
        "consensus": {"source_counts": source_counts,
                      "records": sorted(consensus_list, key=lambda entry: entry["item_id"])},
        "error": error,
    }
    result["conclusion"] = _conclusion(validation, review, error)
    return _round(result)


def dump_json(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return str(value)


def _table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(
            str(cell).replace("|", "\\|") for cell in row) + " |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    run, metric = report["run"], report["metrics"]
    lines = [
        "# 标注一致性核验报告", "",
        f"- 结论：**{report['conclusion']['label']}**（{report['conclusion']['text']}）",
        f"- run_id：`{run['run_id']}`", f"- 开始时间：{run['started_at']}",
        f"- 完成时间：{run['completed_at'] or '-'}",
        f"- 方案指纹：`{report['config']['fingerprint']}`",
        f"- 数据 SHA-256：`{report['data']['file_sha256']}`",
        f"- 有效记录：{report['data']['record_count']}；条目：{report['data']['items']}；"
        f"标注人：{len(report['data']['annotators'])}", "", "## 校验结论", "",
        f"状态 `{report['validation']['status']}`；"
        f"error {report['validation']['error_count']}，"
        f"warning {report['validation']['warning_count']}，"
        f"promoted {report['validation']['promoted_count']}。", "",
    ]
    findings = report["validation"]["findings"]
    if findings:
        lines.extend(_table(
            ["severity", "code", "item", "annotator", "line", "message"],
            ([item["severity"], item["code"], _fmt(item["item_id"]),
              _fmt(item["annotator_id"]), _fmt(item["line_no"]), item["message"]]
             for item in findings)))
    lines.extend(["", "## 一致性度量", ""])
    if metric is None:
        lines.append("本次运行未产出度量。")
    else:
        lines.extend(_table(
            ["metric", "value", "po", "pe", "Do", "De", "items", "annotations"],
            [[metric["name"], _fmt(metric["value"]), _fmt(metric["po"]),
              _fmt(metric["pe"]), _fmt(metric["Do"]), _fmt(metric["De"]),
              metric["eligible_items"], metric["eligible_annotations"]]]))
        if metric["per_label"]:
            lines.extend(["", "### 分标签与宏平均", "",
                          "macro_average = " + _fmt(metric["macro_average"]), ""])
            lines.extend(_table(
                ["label", "value", "po", "pe", "error"],
                ([item["label"], _fmt(item["value"]), _fmt(item["po"]),
                  _fmt(item["pe"]), _fmt(item["error_code"])]
                 for item in metric["per_label"])))
    lines.extend(["", "## 复核队列", ""])
    if report["review_queue"]:
        lines.extend(_table(
            ["item", "reason", "status", "votes", "rate", "annotators"],
            ([item["item_id"], item["reason"], item["status"],
              json.dumps(item["vote_counts"], ensure_ascii=False, sort_keys=True),
              _fmt(item["vote_rate"]), item["annotator_count"]]
             for item in report["review_queue"])))
    else:
        lines.append("无。")
    lines.extend(["", "## 标注人", ""])
    if report["annotators"]:
        lines.extend(_table(
            ["annotator", "agreement", "compared", "participated", "missing",
             "out_of_space", "outlier"],
            ([item["annotator_id"], _fmt(item["agreement_rate"]),
              item["compared_items"], item["participation_items"],
              item["missing_count"], item["out_of_space_count"], item["is_outlier"]]
             for item in report["annotators"])))
    lines.extend(["", "## 留一影响度", ""])
    if report["leave_one_out"]:
        lines.extend(_table(
            ["annotator", "value_without", "delta", "error"],
            ([item["annotator_id"], _fmt(item["value"]), _fmt(item["delta"]),
              _fmt(item["error_code"])] for item in report["leave_one_out"])))
    lines.extend([
        "", "## 裁决与共识输出", "",
        f"裁决文件条目 {report['adjudications']['provided']}；"
        f"采纳 {report['adjudications']['adopted']}；"
        f"解决复核条目 {report['adjudications']['resolved_review_items']}。",
        "共识来源统计：`" +
        json.dumps(report["consensus"]["source_counts"], sort_keys=True) + "`。", "",
    ])
    if report["error"]:
        lines.extend(["## 错误", "",
                      f"`{report['error']['code']}`：{report['error']['message']}", ""])
    lines.extend(["---", "本报告由本地离线的标注合议台生成；"
                  "一致性指标是统计量，不替代人工判断。", ""])
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "report_json": ("report.json", dump_json(report)),
        "report_md": ("report.md", render_markdown(report)),
        "report_html": ("report.html", render_html(report)),
    }
    artifacts = []
    for kind, (name, text) in contents.items():
        path = output_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        artifacts.append(_artifact(kind, path))
    return artifacts


def write_consensus(records: Iterable[ConsensusRecord], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "consensus.jsonl"
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
             for record in sorted(records, key=lambda item: item.item_id)]
    path.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8", newline="\n")
    return _artifact("consensus", path)


def _artifact(kind: str, path: Path) -> dict[str, Any]:
    return {"kind": kind, "path": str(path.resolve()),
            "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
