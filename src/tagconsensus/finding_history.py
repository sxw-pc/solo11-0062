"""Cross-run tracking of validation findings: new, persisting and resolved.

Statuses are computed only when both runs carry finding identity material
(``finding_key`` per finding, ``finding_scope_fingerprint`` per run) and the
validation-relevant plan semantics are identical. Anything else is reported
as incomparable instead of guessing which problems were fixed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ledger import get_run, list_runs

STATUSES = ("new", "persisting", "resolved")


def _finding_map(findings: list[dict]) -> dict[str, dict]:
    mapped: dict[str, dict] = {}
    for item in findings:
        key = item.get("finding_key")
        if key is not None:
            mapped.setdefault(key, item)
    return mapped


def _stored_row(item: dict | None) -> Any:
    """Normalized source row, available only for runs stored with persist_raw_rows."""
    if item is None:
        return None
    raw = item.get("row_json")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


def _entry(current: dict | None, previous: dict | None) -> dict[str, Any]:
    source = current if current is not None else previous or {}
    both = current is not None and previous is not None
    current_line = current.get("line_no") if current is not None else None
    previous_line = previous.get("line_no") if previous is not None else None
    current_content = current.get("content_key") if current is not None else None
    previous_content = previous.get("content_key") if previous is not None else None
    current_row = current.get("row_digest") if current is not None else None
    previous_row = previous.get("row_digest") if previous is not None else None
    return {
        "finding_key": source.get("finding_key"),
        "code": source.get("code"),
        "severity": source.get("severity"),
        "item_id": source.get("item_id"),
        "annotator_id": source.get("annotator_id"),
        "message": source.get("message"),
        "current_line_no": current_line,
        "previous_line_no": previous_line,
        "line_changed": bool(both and current_line != previous_line),
        "content_changed": bool(both and current_content != previous_content),
        "row_changed": bool(both and current_row != previous_row),
        "current_row": _stored_row(current),
        "previous_row": _stored_row(previous),
    }


def _sort_entries(entries: list[dict]) -> list[dict]:
    return sorted(
        entries,
        key=lambda item: (
            item["code"] or "",
            item["item_id"] or "",
            item["annotator_id"] or "",
            item["finding_key"] or "",
        ),
    )


def _identity_gap(detail: dict) -> str | None:
    if detail["run"].get("finding_scope_fingerprint") is None:
        return "run has no finding scope fingerprint"
    if any(item.get("finding_key") is None for item in detail["findings"]):
        return "run has findings without finding_key"
    return None


def _incomparable(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "comparable": False,
        "reason": reason,
        "run_id": extra.get("run_id"),
        "against_run_id": extra.get("against_run_id"),
        "detail": extra.get("detail"),
        "data_changed": None,
        "counts": None,
        "new": [],
        "persisting": [],
        "resolved": [],
    }


def _resolve_run(runs: list[dict], run_id: str | None) -> dict | None:
    if run_id is None:
        return runs[0]
    return next((run for run in runs if run["run_id"] == run_id), None)


def compare_findings(
    db_path: Path, run_id: str | None = None, against_run_id: str | None = None
) -> dict[str, Any]:
    """Compare one run's findings against an earlier run in the same ledger."""
    runs = list_runs(db_path)
    if not runs:
        return _incomparable("no_runs")
    current = _resolve_run(runs, run_id)
    if current is None:
        return _incomparable("run_not_found", run_id=run_id)
    index = next(i for i, run in enumerate(runs) if run["run_id"] == current["run_id"])
    if against_run_id is not None:
        previous = next(
            (run for run in runs if run["run_id"] == against_run_id), None
        )
        if previous is None:
            return _incomparable(
                "run_not_found", run_id=current["run_id"],
                against_run_id=against_run_id,
            )
    else:
        previous = runs[index + 1] if index + 1 < len(runs) else None
        if previous is None:
            return _incomparable("no_previous_run", run_id=current["run_id"])
    pair = {"run_id": current["run_id"], "against_run_id": previous["run_id"]}
    current_detail = get_run(db_path, current["run_id"])
    previous_detail = get_run(db_path, previous["run_id"])
    if current_detail is None or previous_detail is None:
        return _incomparable("run_not_found", **pair)
    gap = _identity_gap(current_detail)
    if gap is not None:
        return _incomparable(
            "missing_identity_material", detail=f"current run: {gap}", **pair
        )
    gap = _identity_gap(previous_detail)
    if gap is not None:
        return _incomparable(
            "missing_identity_material", detail=f"previous run: {gap}", **pair
        )
    current_scope = current_detail["run"]["finding_scope_fingerprint"]
    previous_scope = previous_detail["run"]["finding_scope_fingerprint"]
    if current_scope != previous_scope:
        return _incomparable(
            "config_incompatible",
            detail="validation-relevant plan semantics differ between runs",
            **pair,
        )
    current_map = _finding_map(current_detail["findings"])
    previous_map = _finding_map(previous_detail["findings"])
    new = [_entry(current_map[key], None) for key in current_map.keys() - previous_map]
    resolved = [_entry(None, previous_map[key]) for key in previous_map.keys() - current_map]
    persisting = [
        _entry(current_map[key], previous_map[key])
        for key in current_map.keys() & previous_map
    ]
    counts = {
        "new": len(new),
        "persisting": len(persisting),
        "resolved": len(resolved),
        "line_changed": sum(1 for item in persisting if item["line_changed"]),
        "content_changed": sum(1 for item in persisting if item["content_changed"]),
        "row_changed": sum(1 for item in persisting if item["row_changed"]),
    }
    return {
        "comparable": True,
        "reason": None,
        "detail": None,
        **pair,
        "data_changed": (
            current_detail["run"].get("data_sha256")
            != previous_detail["run"].get("data_sha256")
        ),
        "counts": counts,
        "new": _sort_entries(new),
        "persisting": _sort_entries(persisting),
        "resolved": _sort_entries(resolved),
    }
