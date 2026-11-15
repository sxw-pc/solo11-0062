"""Load reviewer decisions, bind verifiable evidence and merge ahead of voting.

Binding scope (方案B / scheme B): every decision is bound to (1) a per-item
summary of the annotators and labels that make up the item's voting evidence
and (2) a summary of every plan field that can change an item's judgment. A
change inside one item only invalidates that item's decision; edits to an
unrelated item never invalidate another item's decision. A change to a
decision-affecting plan field invalidates every bound decision. Decisions that
no longer match the current evidence are never applied on top of majority
voting and are reported as stale pending re-review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .errors import AdjudicationError
from .hashing import sha256_text, stable_json
from .models import Adjudication, ConsensusRecord, Dataset, ItemConsensus

# Bump when the on-disk evidence binding layout changes.
BINDING_SCHEMA_VERSION = "1.0"
BINDING_SCOPE = "per_item"

# Plan fields that participate in per-item judgment (see items.build_items).
# Deliberately excluded: metric/pair/outlier_threshold/annotators/strict and
# output paths, which never change an item's majority label or review reasons.
DECISION_SCHEME_FIELDS = (
    "label_space",
    "multi_label",
    "min_annotators_per_item",
    "review_threshold",
)

# Stale reason codes (deterministic order, used for sorting too).
STALE_REASON_ORDER = (
    "ADJ_EVIDENCE_MISSING",
    "ADJ_ITEM_EVIDENCE_CHANGED",
    "ADJ_SCHEME_CHANGED",
    "ADJ_BINDING_TAMPERED",
    "ADJ_LABEL_OUT_OF_SPACE",
)


@dataclass(frozen=True)
class EvidenceBasis:
    """Verifiable evidence handed from the CLI run to adjudication handling."""

    binding_schema_version: str
    binding_scope: str
    item_evidence: dict[str, dict[str, Any]]
    item_fingerprints: dict[str, str]
    scheme_summary: dict[str, Any]
    scheme_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_schema_version": self.binding_schema_version,
            "binding_scope": self.binding_scope,
            "scheme_fields": list(DECISION_SCHEME_FIELDS),
            "scheme_summary": self.scheme_summary,
            "scheme_fingerprint": self.scheme_fingerprint,
            "items": {
                item_id: {
                    "summary": self.item_evidence[item_id],
                    "fingerprint": self.item_fingerprints[item_id],
                }
                for item_id in sorted(self.item_evidence)
            },
        }


def decision_scheme_summary(config: Config) -> dict[str, Any]:
    """Plan fields that can change an item's majority label or review state."""
    return {
        "label_space": list(config.label_space),
        "multi_label": config.multi_label,
        "min_annotators_per_item": config.min_annotators_per_item,
        "review_threshold": config.review_threshold,
    }


def _item_vote_summary(dataset: Dataset, item_id: str) -> dict[str, Any]:
    """Every annotation row for one item: annotators and their labels.

    Missing (empty-label) annotations are kept explicitly so a revision that
    turns a vote into a missing annotation changes the fingerprint too.
    """
    rows = sorted(
        (row for row in dataset.annotations if row.item_id == item_id),
        key=lambda row: (row.annotator_id, row.line_no),
    )
    return {
        "item_id": item_id,
        "votes": [
            {"annotator_id": row.annotator_id, "labels": list(row.labels)}
            for row in rows
        ],
    }


def build_evidence_basis(dataset: Dataset, config: Config) -> EvidenceBasis:
    """Compute the current per-item and plan evidence for a whole run."""
    item_evidence: dict[str, dict[str, Any]] = {}
    item_fingerprints: dict[str, str] = {}
    for item_id in dataset.item_ids:
        summary = _item_vote_summary(dataset, item_id)
        item_evidence[item_id] = summary
        item_fingerprints[item_id] = sha256_text(stable_json(summary))
    scheme_summary = decision_scheme_summary(config)
    scheme_fingerprint = sha256_text(stable_json(scheme_summary))
    return EvidenceBasis(
        binding_schema_version=BINDING_SCHEMA_VERSION,
        binding_scope=BINDING_SCOPE,
        item_evidence=item_evidence,
        item_fingerprints=item_fingerprints,
        scheme_summary=scheme_summary,
        scheme_fingerprint=scheme_fingerprint,
    )


def _value(raw: dict, field: str, line_no: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AdjudicationError(
            "ADJ_MISSING_FIELD", f"line {line_no}: {field} is required", field
        )
    return value.strip()


def _fingerprint(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value.strip()) != 64:
        return ""
    return value.strip()


def _parse_evidence(raw: Any, line_no: int) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise AdjudicationError(
            "ADJ_MALFORMED_ROW",
            f"line {line_no}: evidence must be an object",
            "evidence",
        )
    return raw


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
    allowed = {label.lower() for label in config.label_space}
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
        evidence = _parse_evidence(raw.get("evidence"), line_no)
        # A label outside the current label space is only a hard error when the
        # decision carries no binding; bound decisions become stale instead so
        # the reviewer, reason and old decision stay traceable.
        if evidence is None and label not in allowed:
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
        decisions.append(
            Adjudication(item_id, label, reviewer, reason, line_no, evidence=evidence)
        )
    decisions.sort(key=lambda item: item.item_id)
    return decisions


def _embedded_fingerprints(
    evidence: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None, dict[str, Any] | None]:
    item_block = evidence.get("item")
    scheme_block = evidence.get("scheme")
    if not isinstance(item_block, dict) or not isinstance(scheme_block, dict):
        return None, None, None, None, None
    item_fingerprint = _fingerprint(item_block.get("fingerprint"))
    item_summary = item_block.get("summary")
    scheme_fingerprint = _fingerprint(scheme_block.get("fingerprint"))
    scheme_summary = scheme_block.get("summary")
    return (
        item_fingerprint,
        scheme_fingerprint,
        (item_summary if isinstance(item_summary, dict) else None),
        (scheme_summary if isinstance(scheme_summary, dict) else None),
        scheme_block if isinstance(scheme_block, dict) else None,
    )


def verify_adjudications(
    adjudications: Iterable[Adjudication],
    basis: EvidenceBasis,
    config: Config,
) -> list[Adjudication]:
    """Check each decision's binding against the current run evidence.

    Only decisions whose per-item evidence and decision-affecting plan
    semantics still match are returned with status ``valid``. Every other
    decision keeps reviewer/reason/old label and gains explicit stale reasons;
    stale decisions never override the current majority result.
    """
    allowed = {label.lower() for label in config.label_space}
    verified: list[Adjudication] = []
    for decision in adjudications:
        reasons: list[str] = []
        evidence = decision.evidence
        if evidence is None:
            reasons.append("ADJ_EVIDENCE_MISSING")
        else:
            (
                bound_item_fp,
                bound_scheme_fp,
                bound_item_summary,
                bound_scheme_summary,
                _scheme_block,
            ) = _embedded_fingerprints(evidence)
            if (
                bound_item_fp is None
                or bound_scheme_fp is None
                or bound_item_summary is None
                or bound_scheme_summary is None
                or not isinstance(evidence.get("binding_schema_version"), str)
                or evidence.get("binding_scope") != BINDING_SCOPE
            ):
                reasons.append("ADJ_BINDING_TAMPERED")
            else:
                # Recompute the fingerprints from the embedded summaries. A
                # rewritten summary with a stale hash is detected as tampering.
                if (
                    sha256_text(stable_json(bound_item_summary)) != bound_item_fp
                    or sha256_text(stable_json(bound_scheme_summary)) != bound_scheme_fp
                ):
                    reasons.append("ADJ_BINDING_TAMPERED")
                else:
                    current_item_fp = basis.item_fingerprints.get(decision.item_id)
                    if current_item_fp is None or bound_item_fp != current_item_fp:
                        reasons.append("ADJ_ITEM_EVIDENCE_CHANGED")
                    if bound_scheme_fp != basis.scheme_fingerprint:
                        reasons.append("ADJ_SCHEME_CHANGED")
                    # The label must remain inside the *current* label space;
                    # a bound decision whose label was removed cannot be
                    # applied even if its item votes never changed.
                    if decision.decision_label not in allowed:
                        reasons.append("ADJ_LABEL_OUT_OF_SPACE")
        ordered_reasons = tuple(
            code for code in STALE_REASON_ORDER if code in set(reasons)
        )
        if ordered_reasons:
            verified.append(
                Adjudication(
                    decision.item_id,
                    decision.decision_label,
                    decision.reviewer,
                    decision.reason,
                    decision.line_no,
                    evidence=decision.evidence,
                    status="stale",
                    stale_reasons=ordered_reasons,
                )
            )
        else:
            verified.append(
                Adjudication(
                    decision.item_id,
                    decision.decision_label,
                    decision.reviewer,
                    decision.reason,
                    decision.line_no,
                    evidence=decision.evidence,
                    status="valid",
                    stale_reasons=(),
                )
            )
    verified.sort(key=lambda item: item.item_id)
    return verified


def apply_adjudications(
    items: Iterable[ItemConsensus], adjudications: Iterable[Adjudication]
) -> list[ConsensusRecord]:
    decisions = {item.item_id: item for item in adjudications}
    records: list[ConsensusRecord] = []
    for item in sorted(items, key=lambda entry: entry.item_id):
        decision = decisions.get(item.item_id)
        if decision is not None and decision.is_valid:
            records.append(
                ConsensusRecord(
                    item.item_id,
                    (decision.decision_label,),
                    "adjudicated",
                    item.vote_rate,
                    item.review_reasons,
                    decision.reviewer,
                    adjudication_status="valid",
                )
            )
        elif decision is not None:
            # Stale binding: never silently downgrade to a majority vote and
            # never count the old decision as resolved. Labels stay empty so
            # the record cannot be consumed as a usable consensus label; the
            # old decision is preserved alongside the invalidation reasons.
            records.append(
                ConsensusRecord(
                    item.item_id,
                    (),
                    "adjudication_stale",
                    item.vote_rate,
                    item.review_reasons,
                    None,
                    adjudication_status="stale",
                    stale_reasons=decision.stale_reasons,
                    prior_decision_label=decision.decision_label,
                    prior_reviewer=decision.reviewer,
                    prior_reason=decision.reason,
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


def _bound_row(
    decision: Adjudication, basis: EvidenceBasis
) -> dict[str, Any]:
    return {
        "item_id": decision.item_id,
        "decision_label": decision.decision_label,
        "reviewer": decision.reviewer,
        "reason": decision.reason,
        "evidence": {
            "binding_schema_version": basis.binding_schema_version,
            "binding_scope": basis.binding_scope,
            "item": {
                "summary": basis.item_evidence[decision.item_id],
                "fingerprint": basis.item_fingerprints[decision.item_id],
            },
            "scheme": {
                "fields": list(DECISION_SCHEME_FIELDS),
                "summary": basis.scheme_summary,
                "fingerprint": basis.scheme_fingerprint,
            },
        },
    }


def write_bound_adjudications(
    path: Path, adjudications: Iterable[Adjudication], basis: EvidenceBasis
) -> Path:
    """Offline helper: (re)write a decision file with current evidence bound."""
    rows = [
        _bound_row(decision, basis)
        for decision in sorted(adjudications, key=lambda item: item.item_id)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
        newline="\n",
    )
    return path
