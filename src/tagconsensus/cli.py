"""Command line entry points for the offline consensus pipeline."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adjudicate import apply_adjudications, load_adjudications
from .config import Config, load_config
from .errors import TagConsensusError
from .items import annotator_stats, build_items, leave_one_out
from .ledger import list_runs, record_run
from .loader import load_dataset
from .metrics import compute_metric
from .models import Adjudication, ConsensusRecord, Dataset
from .report import build_report, write_consensus, write_reports
from .validate import ValidationResult, validate_dataset


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _decision_records(adjudications: list[Adjudication]) -> list[dict[str, Any]]:
    return [
        {"item_id": item.item_id, "decision_label": item.decision_label,
         "reviewer": item.reviewer, "reason": item.reason, "adopted": True}
        for item in adjudications
    ]


def _persist(
    config: Config, dataset: Dataset, validation: ValidationResult, report: dict,
    consensus: list[ConsensusRecord],
) -> None:
    artifacts = write_reports(report, config.output_dir)
    artifacts.append(write_consensus(consensus, config.output_dir))
    record_run(config.db_path, report, artifacts)


def _blocked(
    *, config: Config, dataset: Dataset, validation: ValidationResult,
    run_id: str, started_at: str, error: dict | None, metric=None,
) -> int:
    report = build_report(
        config=config, dataset=dataset, validation=validation, run_id=run_id,
        started_at=started_at, completed_at=_now(), metric=metric, error=error,
    )
    _persist(config, dataset, validation, report, [])
    print(f"run_id={run_id} conclusion={report['conclusion']['label']} "
          f"error={error['code'] if error else 'VALIDATION_BLOCKED'}")
    print(f"output={config.output_dir}")
    return 2


def run_demo(config_path: str) -> int:
    started_at = _now()
    config = load_config(config_path)
    dataset = load_dataset(config.dataset)
    validation = validate_dataset(dataset, config)
    run_id = uuid.uuid4().hex
    if validation.blocked:
        return _blocked(config=config, dataset=dataset, validation=validation,
                        run_id=run_id, started_at=started_at, error=None)
    try:
        decisions = load_adjudications(
            config.adjudication_file, config, dataset.item_ids
        )
        metric = compute_metric(dataset, config)
        items = build_items(dataset, config)
        stats = annotator_stats(dataset, config, items)
        influence = leave_one_out(dataset, config, metric)
        consensus = apply_adjudications(items, decisions)
    except TagConsensusError as exc:
        return _blocked(config=config, dataset=dataset, validation=validation,
                        run_id=run_id, started_at=started_at, error=exc.as_dict())
    report = build_report(
        config=config, dataset=dataset, validation=validation, run_id=run_id,
        started_at=started_at, completed_at=_now(), metric=metric, items=items,
        annotators=stats, leave_one_out=influence,
        adjudications=_decision_records(decisions), consensus=consensus,
    )
    _persist(config, dataset, validation, report, consensus)
    print(f"run_id={run_id} conclusion={report['conclusion']['label']} "
          f"metric={metric.name} value={metric.value:.12g}")
    print(f"records={dataset.records} items={len(dataset.item_ids)} "
          f"review={len(report['review_queue'])} "
          f"adjudicated={report['consensus']['source_counts']['adjudicated']}")
    print(f"output={config.output_dir}")
    return 0


def run_history(db_path: str) -> int:
    runs = list_runs(Path(db_path))
    if not runs:
        print("no runs")
        return 0
    print("run_id\tstarted_at\tmetric\tvalue\tconclusion\trecords")
    for run in runs:
        value = "-" if run["metric_value"] is None else f"{run['metric_value']:.12g}"
        print(
            f"{run['run_id']}\t{run['started_at']}\t{run['metric_name'] or '-'}\t"
            f"{value}\t{run['conclusion_label']}\t{run['data_records']}"
        )
    return 0


def run_serve(config_path: str) -> int:
    config = load_config(config_path)
    from .webapp import create_app

    app = create_app(str(config.db_path))
    print("serving local read-only UI at 127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tagconsensus")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the complete pipeline")
    demo.add_argument("--config", required=True)
    history = subparsers.add_parser("history", help="list ledger runs")
    history.add_argument("--db", default="var/tagconsensus.db")
    serve = subparsers.add_parser("serve", help="start the local read-only viewer")
    serve.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            return run_demo(args.config)
        if args.command == "history":
            return run_history(args.db)
        if args.command == "serve":
            return run_serve(args.config)
    except TagConsensusError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"LOCAL_IO_ERROR: {exc}", file=sys.stderr)
        return 2
    return 1
