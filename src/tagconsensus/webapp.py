"""Local-only, read-only ledger viewer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request

from .ledger import get_run, latest_run, list_runs
from .models import REVIEW_REASONS

RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ALLOWED_QUERIES = {"overview": set(), "review": {"reason"}, "ledger": {"run_id"}}


def _allowed_query(endpoint: str) -> bool:
    allowed = ALLOWED_QUERIES[endpoint]
    return set(request.args) <= allowed


def _json_value(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _enrich(detail: dict[str, Any]) -> dict[str, Any]:
    for item in detail["review_queue"]:
        item["reasons_list"] = _json_value(item.get("reasons"), [])
        item["votes_dict"] = _json_value(item.get("votes"), {})
    for item in detail["annotators"]:
        item["is_outlier"] = bool(item["is_outlier"])
    for item in detail["findings"]:
        item["promoted"] = bool(item["promoted"])
    for item in detail["adjudications"]:
        item["adopted"] = bool(item["adopted"])
    return detail


def create_app(db_path: str) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.update(READ_ONLY=True)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; style-src 'self'; img-src 'self'; base-uri 'none'"
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def overview():
        if not _allowed_query("overview"):
            abort(400, "unknown query parameter")
        latest = latest_run(Path(db_path))
        detail = get_run(Path(db_path), latest["run_id"]) if latest else None
        return render_template("overview.html", latest=latest, detail=detail)

    @app.get("/review")
    def review():
        if not _allowed_query("review"):
            abort(400, "unknown query parameter")
        reason = request.args.get("reason", "")
        if reason and reason not in REVIEW_REASONS:
            abort(400, "unknown reason code")
        latest = latest_run(Path(db_path))
        detail = get_run(Path(db_path), latest["run_id"]) if latest else None
        if detail is not None:
            detail = _enrich(detail)
        entries = []
        if detail is not None:
            entries = [
                item for item in detail["review_queue"]
                if not reason or item.get("reason") == reason
            ]
        return render_template(
            "review.html", latest=latest, detail=detail, entries=entries,
            reason=reason, reasons=REVIEW_REASONS,
        )

    @app.get("/ledger")
    def ledger():
        if not _allowed_query("ledger"):
            abort(400, "unknown query parameter")
        runs = list_runs(Path(db_path))
        run_id = request.args.get("run_id", "")
        detail = None
        if run_id:
            if not RUN_ID_PATTERN.match(run_id):
                abort(400, "invalid run_id")
            detail = get_run(Path(db_path), run_id)
            if detail is None:
                abort(404, "run_id not found")
            detail = _enrich(detail)
        return render_template("ledger.html", runs=runs, detail=detail, run_id=run_id)

    return app
