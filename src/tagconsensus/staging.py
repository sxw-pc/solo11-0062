"""Bounded-memory ingestion via a temporary SQLite stage.

Large CSV/JSONL inputs are normalized one physical line at a time by the
streaming parsers in :mod:`tagconsensus.loader` and spilled into a private
temporary SQLite database. The database holds:

* every valid, normalized annotation with its original physical line number;
* every parse finding (malformed rows, invalid confidence, ...) with the same
  line numbers as the in-memory path;
* a global key index on ``(annotator_id, item_id)`` so duplicate annotations
  are detected across the whole file even when repeats land many batches apart.

Lifecycle and failure semantics:

* all writes happen inside one explicit transaction; any staging failure rolls
  the transaction back and the uniquely-named temporary directory (and only
  that directory) is removed, so a parsed prefix is never exposed as a usable
  dataset and no user file is ever touched;
* validation runs against the staged data before any metric is computed, so a
  malformed or duplicated prefix can never reach the metrics stage;
* the staged view implements the same :class:`~tagconsensus.models.RecordSource`
  surface as the in-memory :class:`~tagconsensus.models.Dataset`, and every
  query orders rows by physical line number, keeping reports item-identical.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .config import DatasetSpec
from .errors import DataError
from .loader import (
    DEFAULT_CHUNK_SIZE,
    ParseEvent,
    csv_events,
    jsonl_events,
)
from .models import Annotation, ValidationFinding

_STAGE_PREFIX = "tagconsensus-stage-"
_INSERT_BATCH = 2000

_SCHEMA = """
CREATE TABLE annotations (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  annotator_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  labels_json TEXT NOT NULL,
  confidence_json TEXT NOT NULL,
  is_missing INTEGER NOT NULL,
  line_no INTEGER NOT NULL
);
CREATE TABLE parse_findings (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL,
  severity TEXT NOT NULL,
  message TEXT NOT NULL,
  item_id TEXT,
  annotator_id TEXT,
  line_no INTEGER,
  promoted INTEGER NOT NULL DEFAULT 0
);
"""

_INDEXES = """
CREATE INDEX idx_annotations_key_line
  ON annotations (annotator_id, item_id, line_no);
CREATE INDEX idx_annotations_line ON annotations (line_no);
"""


class StagedDataset:
    """A :class:`RecordSource` backed by a private temporary SQLite database."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        temp_dir: Path,
        *,
        source_path: str,
        file_format: str,
        file_sha256: str,
        record_count: int,
        parse_finding_count: int,
        missing_count: int,
        item_ids: tuple[str, ...],
        annotators: tuple[str, ...],
        owner: bool = True,
        excluded_annotator: str | None = None,
    ) -> None:
        self._connection = connection
        self._temp_dir = temp_dir
        self._source_path = source_path
        self._file_format = file_format
        self._file_sha256 = file_sha256
        self._record_count = record_count
        self._parse_finding_count = parse_finding_count
        self._missing_count = missing_count
        self._item_ids = item_ids
        self._annotators = annotators
        self._owner = owner
        self._excluded_annotator = excluded_annotator
        self._closed = False

    # -- RecordSource surface ---------------------------------------------

    @property
    def records(self) -> int:
        return self._record_count

    @property
    def source_path(self) -> str:
        return self._source_path

    @property
    def file_format(self) -> str:
        return self._file_format

    @property
    def file_sha256(self) -> str:
        return self._file_sha256

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self._item_ids

    @property
    def annotators(self) -> tuple[str, ...]:
        return self._annotators

    @property
    def missing_annotations(self) -> int:
        return self._missing_count

    @property
    def annotator_fingerprint(self) -> str:
        from .hashing import annotator_fingerprint

        return annotator_fingerprint(self._annotators)

    def _row_params(self) -> tuple:
        return (self._excluded_annotator,) if self._excluded_annotator is not None else ()

    def _row_where(self) -> str:
        return (
            "WHERE annotator_id != ?" if self._excluded_annotator is not None else ""
        )

    def iter_annotations(self) -> Iterator[Annotation]:
        self._ensure_open()
        query = (
            "SELECT annotator_id, item_id, labels_json, confidence_json, line_no "
            "FROM annotations "
            + self._row_where()
            + " ORDER BY line_no, row_id"
        )
        for annotator_id, item_id, labels_json, confidence_json, line_no in (
            self._connection.execute(query, self._row_params())
        ):
            yield Annotation(
                annotator_id,
                item_id,
                tuple(json.loads(labels_json)),
                json.loads(confidence_json),
                line_no,
            )

    def iter_parse_findings(self) -> Iterator[ValidationFinding]:
        self._ensure_open()
        query = (
            "SELECT code, severity, message, item_id, annotator_id, line_no, promoted "
            "FROM parse_findings ORDER BY row_id"
        )
        for code, severity, message, item_id, annotator_id, line_no, promoted in (
            self._connection.execute(query)
        ):
            yield ValidationFinding(
                code,
                severity,
                message,
                item_id=item_id,
                annotator_id=annotator_id,
                line_no=line_no,
                promoted=bool(promoted),
            )

    def iter_duplicate_groups(
        self,
    ) -> Iterator[tuple[tuple[str, str], list[int]]]:
        """Global duplicate detection via the key index, not per-chunk state."""
        self._ensure_open()
        query = (
            "SELECT annotator_id, item_id, line_no FROM annotations "
            + self._row_where()
            + " ORDER BY annotator_id, item_id, line_no, row_id"
        )
        current_key: tuple[str, str] | None = None
        current_lines: list[int] = []
        for annotator_id, item_id, line_no in self._connection.execute(
            query, self._row_params()
        ):
            key = (annotator_id, item_id)
            if key != current_key:
                if current_key is not None and len(current_lines) >= 2:
                    yield current_key, current_lines
                current_key = key
                current_lines = []
            current_lines.append(line_no)
        if current_key is not None and len(current_lines) >= 2:
            yield current_key, current_lines

    def without_annotator(self, annotator_id: str) -> "StagedDataset":
        """Return a filtered view sharing the same stage (no data copy)."""
        self._ensure_open()
        kept_annotators = tuple(
            name for name in self._annotators if name != annotator_id
        )
        record_count = self._connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE annotator_id != ?",
            (annotator_id,),
        ).fetchone()[0]
        missing_count = self._connection.execute(
            "SELECT COUNT(*) FROM annotations "
            "WHERE annotator_id != ? AND is_missing = 1",
            (annotator_id,),
        ).fetchone()[0]
        item_rows = self._connection.execute(
            "SELECT DISTINCT item_id FROM annotations "
            "WHERE annotator_id != ? ORDER BY item_id",
            (annotator_id,),
        ).fetchall()
        item_ids = tuple(row[0] for row in item_rows)
        return StagedDataset(
            self._connection,
            self._temp_dir,
            source_path=self._source_path,
            file_format=self._file_format,
            file_sha256=self._file_sha256,
            record_count=record_count,
            parse_finding_count=self._parse_finding_count,
            missing_count=missing_count,
            item_ids=item_ids,
            annotators=kept_annotators,
            owner=False,
            excluded_annotator=annotator_id,
        )

    def fingerprint_dict(self) -> dict[str, Any]:
        return {
            "file_sha256": self._file_sha256,
            "record_count": self._record_count,
            "annotator_fingerprint": self.annotator_fingerprint,
        }

    def fingerprint(self) -> str:
        from .hashing import sha256_text, stable_json

        return sha256_text(stable_json(self.fingerprint_dict()))

    # -- lifecycle ---------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise DataError(
                "DATA_STAGING_CLOSED",
                "staged dataset has already been released",
                "dataset.path",
            )

    def close(self) -> None:
        """Release the temporary database and its private directory."""
        if self._closed:
            return
        self._closed = True
        if not self._owner:
            # Filtered views share the owner's connection and temp directory.
            return
        try:
            self._connection.close()
        finally:
            # Only the uniquely-named directory created for this run is removed.
            _remove_temp_dir(self._temp_dir)

    def __enter__(self) -> "StagedDataset":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort safety net
        try:
            self.close()
        except Exception:
            pass


def _remove_temp_dir(temp_dir: Path) -> None:
    """Remove only the uniquely-named temp directory created for this run."""
    try:
        shutil.rmtree(temp_dir, ignore_errors=False)
    except FileNotFoundError:
        pass
    except OSError:
        # A private, uniquely-named temp dir left behind on a hostile
        # filesystem must not crash the pipeline or touch any user data.
        pass


def _connect(stage_db: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(stage_db))
    # The stage is ephemeral: trade durability for bounded, small memory use.
    connection.execute("PRAGMA journal_mode = MEMORY")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -2000")  # ~2 MiB page cache
    return connection


def stage_dataset(
    spec: DatasetSpec,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    base_dir: Path | None = None,
) -> StagedDataset:
    """Stream ``spec`` into a temporary SQLite stage with bounded memory.

    Raises :class:`DataError` with a stable code for missing files, encoding
    failures and staging I/O errors; on every failure the partial transaction
    is rolled back and the private temp directory is deleted.
    """
    if spec.format == "csv":
        event_factory = csv_events
        file_format = "csv"
    elif spec.format == "jsonl":
        event_factory = jsonl_events
        file_format = "jsonl"
    else:
        raise DataError(
            "DATA_FORMAT", f"unsupported format {spec.format}", "dataset.format"
        )

    temp_dir: Path | None = None
    connection: sqlite3.Connection | None = None
    try:
        try:
            parent = None if base_dir is None else str(base_dir)
            temp_dir = Path(tempfile.mkdtemp(prefix=_STAGE_PREFIX, dir=parent))
            stage_db = temp_dir / "stage.db"
            connection = _connect(stage_db)
            connection.execute("BEGIN")
            connection.executescript(_SCHEMA)
        except (OSError, sqlite3.Error) as exc:
            raise DataError(
                "DATA_STAGING",
                f"cannot create staging database: {exc}",
                "dataset.path",
            ) from exc

        hasher = hashlib.sha256()
        record_rows: list[tuple] = []
        finding_rows: list[tuple] = []
        record_count = 0
        missing_count = 0
        item_ids: set[str] = set()
        annotators: set[str] = set()
        finding_count = 0
        try:
            events: Iterator[ParseEvent] = event_factory(
                spec.path, spec.encoding, hasher, chunk_size
            )
            for event in events:
                if event.annotation is not None:
                    row = event.annotation
                    record_count += 1
                    if row.is_missing:
                        missing_count += 1
                    item_ids.add(row.item_id)
                    annotators.add(row.annotator_id)
                    record_rows.append(
                        (
                            row.annotator_id,
                            row.item_id,
                            json.dumps(list(row.labels), ensure_ascii=False),
                            json.dumps(row.confidence),
                            1 if row.is_missing else 0,
                            row.line_no,
                        )
                    )
                if event.finding is not None:
                    finding = event.finding
                    finding_count += 1
                    finding_rows.append(
                        (
                            finding.code,
                            finding.severity,
                            finding.message,
                            finding.item_id,
                            finding.annotator_id,
                            finding.line_no,
                            1 if finding.promoted else 0,
                        )
                    )
                if len(record_rows) >= _INSERT_BATCH or len(finding_rows) >= _INSERT_BATCH:
                    _flush(connection, record_rows, finding_rows)
                    record_rows = []
                    finding_rows = []
            _flush(connection, record_rows, finding_rows)
            connection.executescript(_INDEXES)
            connection.commit()
        except UnicodeError:
            # Encoding errors are already wrapped as DataError by the parser;
            # keep any other Unicode failure on the stable encoding code too.
            connection.rollback()
            raise
        except DataError:
            connection.rollback()
            raise
        except (OSError, sqlite3.Error) as exc:
            connection.rollback()
            raise DataError(
                "DATA_STAGING",
                f"cannot write staging database: {exc}",
                "dataset.path",
            ) from exc

        return StagedDataset(
            connection,
            temp_dir,
            source_path=spec.raw_path,
            file_format=file_format,
            file_sha256=hasher.hexdigest(),
            record_count=record_count,
            parse_finding_count=finding_count,
            missing_count=missing_count,
            item_ids=tuple(sorted(item_ids)),
            annotators=tuple(sorted(annotators)),
        )
    except Exception:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        if temp_dir is not None:
            _remove_temp_dir(temp_dir)
        raise


def _flush(
    connection: sqlite3.Connection,
    record_rows: list[tuple],
    finding_rows: list[tuple],
) -> None:
    if record_rows:
        connection.executemany(
            "INSERT INTO annotations "
            "(annotator_id, item_id, labels_json, confidence_json, is_missing, line_no) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            record_rows,
        )
    if finding_rows:
        connection.executemany(
            "INSERT INTO parse_findings "
            "(code, severity, message, item_id, annotator_id, line_no, promoted) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            finding_rows,
        )
