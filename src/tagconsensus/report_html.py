"""Self-contained HTML rendering for the archived report."""

from __future__ import annotations

import html
import json
from typing import Any, Iterable

HTML_STYLE = ("body{font-family:Georgia,serif;margin:2rem auto;max-width:1100px;padding:0 1rem;"
              "background:#f7f4ed;color:#20201d;line-height:1.5}h1,h2{font-family:Arial,sans-serif;"
              "color:#183a37}table{border-collapse:collapse;width:100%;margin:.6rem 0 1.2rem}"
              "th,td{border:1px solid #c9c2b5;padding:.35rem .5rem;text-align:left;"
              "vertical-align:top}th{background:#e3ddd0}.badge{display:inline-block;padding:.1rem "
              ".5rem;border-radius:999px;background:#183a37;color:#fff}.blocked{background:#7a2929}"
              ".review{background:#8a5a12}.pass{background:#2c5f2d}code{background:#ece7dc;"
              "padding:.05rem .25rem}.muted{color:#5b584f}")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.12f}".rstrip("0").rstrip(".")
    return str(value)


def _html_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join("<tr>" + "".join(
        f"<td>{html.escape(_fmt(cell))}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: dict[str, Any]) -> str:
    conclusion, metric = report["conclusion"], report["metrics"]
    parts = [
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        f"<title>标注一致性核验报告 {html.escape(report['run']['run_id'])}</title>",
        f"<style>{HTML_STYLE}</style></head><body>",
        "<h1>标注一致性核验报告</h1>",
        f"<p><span class=\"badge {conclusion['code']}\">"
        f"{html.escape(conclusion['label'])}</span> {html.escape(conclusion['text'])}</p>",
        f"<p class=\"muted\">run_id <code>{html.escape(report['run']['run_id'])}</code> · "
        f"{html.escape(report['run']['started_at'])}</p>", "<h2>数据与校验</h2>",
        _html_table(["field", "value"], [
            ("file_sha256", report["data"]["file_sha256"]),
            ("record_count", report["data"]["record_count"]),
            ("items", report["data"]["items"]),
            ("annotator_fingerprint", report["data"]["annotator_fingerprint"]),
            ("validation_status", report["validation"]["status"]),
            ("errors", report["validation"]["error_count"]),
            ("warnings", report["validation"]["warning_count"])]),
        "<h2>一致性度量</h2>",
    ]
    if metric is None:
        parts.append("<p>本次运行未产出度量。</p>")
    else:
        parts.append(_html_table(
            ["metric", "value", "po", "pe", "Do", "De", "items", "annotations"],
            [(metric["name"], metric["value"], metric["po"], metric["pe"], metric["Do"],
              metric["De"], metric["eligible_items"], metric["eligible_annotations"])]))
        if metric["per_label"]:
            parts.append("<h3>分标签与宏平均</h3><p>macro_average = "
                         f"{html.escape(_fmt(metric['macro_average']))}</p>")
            parts.append(_html_table(
                ["label", "value", "po", "pe", "error"],
                [(item["label"], item["value"], item["po"], item["pe"],
                  item["error_code"]) for item in metric["per_label"]]))
    parts.append("<h2>复核队列</h2>")
    if report["review_queue"]:
        parts.append(_html_table(
            ["item", "reason", "status", "votes", "rate", "annotators"],
            [(item["item_id"], item["reason"], item["status"],
              json.dumps(item["vote_counts"], ensure_ascii=False, sort_keys=True),
              item["vote_rate"], item["annotator_count"])
             for item in report["review_queue"]]))
    else:
        parts.append("<p>无。</p>")
    parts.extend([
        "<h2>标注人</h2>",
        _html_table(["annotator", "agreement", "compared", "participated", "missing",
                     "out_of_space", "outlier"],
                    [(item["annotator_id"], item["agreement_rate"],
                      item["compared_items"], item["participation_items"],
                      item["missing_count"], item["out_of_space_count"],
                      item["is_outlier"]) for item in report["annotators"]]),
        "<h2>留一影响度</h2>",
        _html_table(["annotator", "value_without", "delta", "error"],
                    [(item["annotator_id"], item["value"], item["delta"],
                      item["error_code"]) for item in report["leave_one_out"]]),
        "<h2>裁决与共识</h2>",
        f"<p>裁决 {report['adjudications']['provided']} 条，采纳 "
        f"{report['adjudications']['adopted']} 条，解决复核 "
        f"{report['adjudications']['resolved_review_items']} 条。共识来源：<code>"
        f"{html.escape(json.dumps(report['consensus']['source_counts'], sort_keys=True))}"
        "</code>。</p>",
    ])
    if report["error"]:
        parts.append(f"<h2>错误</h2><p><code>{html.escape(report['error']['code'])}</code> "
                     f"{html.escape(report['error']['message'])}</p>")
    parts.append("<p class=\"muted\">本报告由本地离线的标注合议台生成；"
                 "一致性指标是统计量，不替代人工判断。</p></body></html>")
    return "".join(parts)
