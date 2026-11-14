"""Load reviewer decisions and merge them ahead of majority voting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import Config
from .errors import AdjudicationError
from .models import Adjudication, ConsensusRecord, ItemConsensus


def _value(raw: dict, field: str, line_no: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AdjudicationError(
            "ADJ_MISSING_FIELD", f"line {line_no}: {field} is required", field
        )
    return value.strip()


def load_adjudications(
    path: Path | None, config: Config, item_ids: Iterable[str]
) -> list[Adjudication]:
    if path is None:
        return []
    if not path.exists() or not path.is_file():
        raise AdjudicationError(
            "ADJ_FILE_NOT_FOUND", f"adjudication file not found: {path}", "adjudication_file"
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise AdjudicationError(
            "ADJ_READ_ERROR", f"cannot read adjudication file: {exc}", "adjudication_file"
        ) from exc
    known_items = set(item_ids)
    allowed = set(config.label_space)
    decisions: list[Adjudication] = []
    seen: dict[str, int] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdjudicationError(
                "ADJ_MALFORMED_ROW", f"line {line_no}: invalid JSON: {exc.msg}", "adjudication"
            ) from exc
        if not isinstance(raw, dict):
            raise AdjudicationError(
                "ADJ_MALFORMED_ROW", f"line {line_no}: row must be an object", "adjudication"
            )
        item_id = _value(raw, "item_id", line_no)
        label = _value(raw, "decision_label", line_no).lower()
        reviewer = _value(raw, "reviewer", line_no)
        reason = _value(raw, "reason", line_no)
        if item_id not in known_items:
            raise AdjudicationError(
                "ADJ_UNKNOWN_ITEM", f"line {line_no}: unknown item {item_id}", "item_id"
            )
        if label not in allowed:
            raise AdjudicationError(
                "ADJ_LABEL_OUT_OF_SPACE",
                f"line {line_no}: label {label} is outside label_space",
                "decision_label",
            )
        if item_id in seen:
            raise AdjudicationError(
                "ADJ_DUPLICATE_ITEM",
                f"line {line_no}: duplicate decision for {item_id} (first line {seen[item_id]})",
                "item_id",
            )
        seen[item_id] = line_no
        decisions.append(Adjudication(item_id, label, reviewer, reason, line_no))
    decisions.sort(key=lambda item: item.item_id)
    return decisions


def apply_adjudications(
    items: Iterable[ItemConsensus], adjudications: Iterable[Adjudication]
) -> list[ConsensusRecord]:
    decisions = {item.item_id: item for item in adjudications}
    records: list[ConsensusRecord] = []
    for item in sorted(items, key=lambda entry: entry.item_id):
        decision = decisions.get(item.item_id)
        if decision is not None:
            records.append(
                ConsensusRecord(
                    item.item_id,
                    (decision.decision_label,),
                    "adjudicated",
                    item.vote_rate,
                    item.review_reasons,
                    decision.reviewer,
                )
            )
        elif item.needs_review:
            records.append(
                ConsensusRecord(
                    item.item_id, (), "unresolved", item.vote_rate, item.review_reasons
                )
            )
        else:
            records.append(
                ConsensusRecord(
                    item.item_id,
                    item.consensus_labels,
                    "majority",
                    item.vote_rate,
                    item.review_reasons,
                )
            )
    return records
