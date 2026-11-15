"""SQLite run ledger: append-only writes from the CLI, SELECT-only reads for the web."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .errors import LedgerError
from .hashing import stable_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, completed_at TEXT,
  config_path TEXT, config_fingerprint TEXT, finding_scope_fingerprint TEXT,
  data_sha256 TEXT, data_records INTEGER,
  annotator_fingerprint TEXT, metric_name TEXT, metric_value REAL,
  conclusion TEXT, conclusion_label TEXT, validation_status TEXT, error_code TEXT,
  report_json_path TEXT, report_md_path TEXT, report_html_path TEXT,
  consensus_path TEXT, output_dir TEXT
);
CREATE TABLE IF NOT EXISTS validation_findings (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, code TEXT NOT NULL,
  severity TEXT NOT NULL, message TEXT NOT NULL, item_id TEXT, annotator_id TEXT,
  line_no INTEGER, promoted INTEGER NOT NULL DEFAULT 0,
  finding_key TEXT, content_key TEXT, row_digest TEXT, row_json TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  run_id TEXT NOT NULL, label TEXT NOT NULL, metric_name TEXT NOT NULL,
  value REAL, po REAL, pe REAL, do_value REAL, de_value REAL,
  eligible_items INTEGER, eligible_annotations INTEGER, error_code TEXT,
  PRIMARY KEY (run_id, label)
);
CREATE TABLE IF NOT EXISTS review_queue (
  run_id TEXT NOT NULL, item_id TEXT NOT NULL, reason TEXT, reasons TEXT,
  annotator_count INTEGER, vote_rate REAL, votes TEXT, status TEXT,
  PRIMARY KEY (run_id, item_id)
);
CREATE TABLE IF NOT EXISTS annotators (
  run_id TEXT NOT NULL, annotator_id TEXT NOT NULL, agreement_rate REAL,
  compared_items INTEGER, participation_items INTEGER, missing_count INTEGER,
  out_of_space_count INTEGER, is_outlier INTEGER,
  PRIMARY KEY (run_id, annotator_id)
);
CREATE TABLE IF NOT EXISTS adjudications (
  run_id TEXT NOT NULL, item_id TEXT NOT NULL, decision_label TEXT,
  reviewer TEXT, reason TEXT, adopted INTEGER,
  PRIMARY KEY (run_id, item_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT NOT NULL, kind TEXT NOT NULL, path TEXT NOT NULL,
  size_bytes INTEGER, sha256 TEXT, PRIMARY KEY (run_id, kind)
);
"""

# Columns added after the first release; old databases gain them via ALTER
# and keep NULLs, which marks their runs as lacking finding identity material.
MIGRATIONS = {
    "runs": ("finding_scope_fingerprint TEXT",),
    "validation_findings": (
        "finding_key TEXT", "content_key TEXT", "row_digest TEXT", "row_json TEXT",
    ),
}


def _migrate(connection: sqlite3.Connection) -> None:
    for table, columns in MIGRATIONS.items():
        existing = {
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        }
        for column in columns:
            if column.split()[0] not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column}")


def init_db(db_path: Path) -> None:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path))
        try:
            connection.executescript(SCHEMA)
            _migrate(connection)
            connection.commit()
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        raise LedgerError("LEDGER_INIT", f"cannot initialize ledger: {exc}", "db_path") from exc


def _insert_many(
    connection: sqlite3.Connection, statement: str, rows: Iterable[tuple]
) -> None:
    values = list(rows)
    if values:
        connection.executemany(statement, values)


def record_run(
    db_path: Path, report: dict[str, Any], artifacts: list[dict[str, Any]]
) -> None:
    init_db(db_path)
    run = report["run"]
    artifact_by_kind = {entry["kind"]: entry for entry in artifacts}
    metrics = report.get("metrics")
    conclusion = report["conclusion"]
    run_row = (
        run["run_id"], run["started_at"], run.get("completed_at"),
        report["config"]["config_path"], report["config"]["fingerprint"],
        report["config"].get("finding_scope_fingerprint"),
        report["data"]["file_sha256"], report["data"]["record_count"],
        report["data"]["annotator_fingerprint"],
        metrics["name"] if metrics else None, metrics["value"] if metrics else None,
        conclusion["code"], conclusion["label"], report["validation"]["status"],
        run.get("error_code"),
        artifact_by_kind.get("report_json", {}).get("path"),
        artifact_by_kind.get("report_md", {}).get("path"),
        artifact_by_kind.get("report_html", {}).get("path"),
        artifact_by_kind.get("consensus", {}).get("path"),
        report["config"]["output_dir"],
    )
    try:
        connection = sqlite3.connect(str(db_path))
        try:
            connection.execute(
                "INSERT OR REPLACE INTO runs VALUES (" + ",".join("?" * 20) + ")", run_row
            )
            for table in (
                "validation_findings", "metrics", "review_queue",
                "annotators", "adjudications", "artifacts",
            ):
                connection.execute(f"DELETE FROM {table} WHERE run_id = ?", (run["run_id"],))
            _insert_many(
                connection,
                "INSERT INTO validation_findings "
                "(run_id,code,severity,message,item_id,annotator_id,line_no,promoted,"
                "finding_key,content_key,row_digest,row_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        run["run_id"], item["code"], item["severity"], item["message"],
                        item["item_id"], item["annotator_id"], item["line_no"],
                        int(bool(item["promoted"])),
                        item.get("finding_key"), item.get("content_key"),
                        item.get("row_digest"),
                        stable_json(item["row"]) if item.get("row") is not None else None,
                    )
                    for item in report["validation"]["findings"]
                ],
            )
            metric_rows: list[tuple] = []
            if metrics:
                metric_rows.append(_metric_row(run["run_id"], "overall", metrics))
                for entry in metrics.get("per_label", []):
                    metric_rows.append(
                        _metric_row(run["run_id"], entry["label"], entry)
                    )
            _insert_many(
                connection,
                "INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?)", metric_rows
            )
            _insert_many(
                connection,
                "INSERT INTO review_queue VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        run["run_id"], item["item_id"], item.get("reason"),
                        stable_json(item.get("review_reasons", [])),
                        item.get("annotator_count"), item.get("vote_rate"),
                        stable_json(item.get("vote_counts", {})), item.get("status"),
                    )
                    for item in report.get("review_queue", [])
                ],
            )
            _insert_many(
                connection,
                "INSERT INTO annotators VALUES (?,?,?,?,?,?,?,?)",
                [
                    (
                        run["run_id"], item["annotator_id"], item["agreement_rate"],
                        item["compared_items"], item["participation_items"],
                        item["missing_count"], item["out_of_space_count"],
                        int(item["is_outlier"]),
                    )
                    for item in report.get("annotators", [])
                ],
            )
            adjudication_items = report.get("adjudications", {}).get("items", [])
            _insert_many(
                connection,
                "INSERT INTO adjudications VALUES (?,?,?,?,?,?)",
                [
                    (
                        run["run_id"], item["item_id"], item["decision_label"],
                        item["reviewer"], item["reason"], int(item["adopted"]),
                    )
                    for item in adjudication_items
                ],
            )
            _insert_many(
                connection,
                "INSERT INTO artifacts VALUES (?,?,?,?,?)",
                [
                    (
                        run["run_id"], item["kind"], item["path"],
                        item["size_bytes"], item["sha256"],
                    )
                    for item in artifacts
                ],
            )
            connection.commit()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise LedgerError("LEDGER_WRITE", f"cannot write ledger: {exc}", "db_path") from exc


def _metric_row(run_id: str, label: str, item: dict[str, Any]) -> tuple:
    return (
        run_id, label, item.get("name", ""), item.get("value"), item.get("po"),
        item.get("pe"), item.get("Do"), item.get("De"), item.get("eligible_items", 0),
        item.get("eligible_annotations", 0), item.get("error_code"),
    )


def _connect_read(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        return None
    try:
        connection = sqlite3.connect(str(db_path))
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error:
        return None


def _rows(connection: sqlite3.Connection, query: str, params: tuple = ()) -> list[dict]:
    try:
        return [dict(row) for row in connection.execute(query, params).fetchall()]
    except sqlite3.Error:
        return []


def list_runs(db_path: Path) -> list[dict[str, Any]]:
    connection = _connect_read(db_path)
    if connection is None:
        return []
    try:
        return _rows(
            connection,
            "SELECT run_id, started_at, completed_at, metric_name, metric_value, "
            "conclusion, conclusion_label, validation_status, data_records, output_dir "
            "FROM runs ORDER BY started_at DESC, rowid DESC",
        )
    finally:
        connection.close()


def latest_run(db_path: Path) -> dict[str, Any] | None:
    runs = list_runs(db_path)
    return runs[0] if runs else None


def get_run(db_path: Path, run_id: str) -> dict[str, Any] | None:
    connection = _connect_read(db_path)
    if connection is None:
        return None
    try:
        run_rows = _rows(connection, "SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not run_rows:
            return None
        return {
            "run": run_rows[0],
            "findings": _rows(
                connection,
                "SELECT * FROM validation_findings WHERE run_id = ? "
                "ORDER BY severity, code, id", (run_id,),
            ),
            "metrics": _rows(
                connection,
                "SELECT * FROM metrics WHERE run_id = ? ORDER BY label", (run_id,),
            ),
            "review_queue": _rows(
                connection,
                "SELECT * FROM review_queue WHERE run_id = ? ORDER BY item_id", (run_id,),
            ),
            "annotators": _rows(
                connection,
                "SELECT * FROM annotators WHERE run_id = ? ORDER BY annotator_id", (run_id,),
            ),
            "adjudications": _rows(
                connection,
                "SELECT * FROM adjudications WHERE run_id = ? ORDER BY item_id", (run_id,),
            ),
            "artifacts": _rows(
                connection,
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY kind", (run_id,),
            ),
        }
    finally:
        connection.close()
