"""Load and strictly validate the declarative consistency plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .assignment import AssignmentManifestSpec, parse_assignment_spec
from .errors import ConfigError
from .hashing import sha256_text, stable_json

METRICS = ("cohen", "fleiss", "krippendorff_nominal", "pairwise_agreement")
FORMATS = ("csv", "jsonl")
ENCODINGS = ("utf-8", "utf-8-sig", "gbk")
TOP_LEVEL = {
    "dataset",
    "label_space",
    "annotators",
    "metric",
    "pair",
    "multi_label",
    "min_annotators_per_item",
    "review_threshold",
    "outlier_threshold",
    "adjudication_file",
    "assignment_manifest",
    "output_dir",
    "db_path",
    "strict",
}
DATASET_FIELDS = {"path", "format", "encoding"}


@dataclass(frozen=True)
class DatasetSpec:
    path: Path
    raw_path: str
    format: str
    encoding: str


@dataclass(frozen=True)
class Config:
    config_path: Path
    dataset: DatasetSpec
    label_space: tuple[str, ...]
    annotators: tuple[str, ...]
    metric: str
    pair: tuple[str, str] | None
    multi_label: bool
    min_annotators_per_item: int
    review_threshold: float
    outlier_threshold: float
    adjudication_file: Path | None
    assignment_manifest: AssignmentManifestSpec | None
    output_dir: Path
    db_path: Path
    strict: bool
    fingerprint: str

    def summary(self) -> dict[str, Any]:
        return {
            "config_path": str(self.config_path),
            "dataset_path": self.dataset.raw_path,
            "dataset_format": self.dataset.format,
            "dataset_encoding": self.dataset.encoding,
            "label_space": list(self.label_space),
            "annotators": list(self.annotators),
            "metric": self.metric,
            "pair": list(self.pair) if self.pair else None,
            "multi_label": self.multi_label,
            "min_annotators_per_item": self.min_annotators_per_item,
            "review_threshold": self.review_threshold,
            "outlier_threshold": self.outlier_threshold,
            "adjudication_file": (
                str(self.adjudication_file) if self.adjudication_file else None
            ),
            "assignment_manifest": _manifest_summary(self.assignment_manifest),
            "output_dir": str(self.output_dir),
            "db_path": str(self.db_path),
            "strict": self.strict,
            "fingerprint": self.fingerprint,
        }


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def _manifest_summary(spec: AssignmentManifestSpec | None) -> dict[str, Any] | None:
    if spec is None:
        return None
    return {"path": str(spec.path), "format": spec.format, "encoding": spec.encoding}


def _required(mapping: dict[str, Any], key: str, field: str) -> Any:
    if key not in mapping:
        raise ConfigError("CFG_MISSING_FIELD", f"missing required field {field}", field)
    return mapping[key]


def _normal_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("CFG_TYPE_INVALID", f"{field} must be a non-empty string", field)
    return value.strip()


def _threshold(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError("CFG_THRESHOLD_RANGE", f"{key} must be a number", key)
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise ConfigError("CFG_THRESHOLD_RANGE", f"{key} must be within [0, 1]", key)
    return number


def _positive_int(mapping: dict[str, Any], key: str, default: int) -> int:
    value = mapping.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ConfigError("CFG_INT_RANGE", f"{key} must be an integer >= 2", key)
    return value


def _string_list(value: Any, field: str, lowercase: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError("CFG_TYPE_INVALID", f"{field} must be a list", field)
    if not value:
        code = "CFG_LABEL_EMPTY" if field == "label_space" else "CFG_TYPE_INVALID"
        raise ConfigError(code, f"{field} must not be empty", field)
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ConfigError(
                "CFG_TYPE_INVALID", f"{field}[{index}] must be a string", field
            )
        text = item.strip()
        if not text:
            code = "CFG_LABEL_EMPTY" if field == "label_space" else "CFG_TYPE_INVALID"
            raise ConfigError(code, f"{field}[{index}] is empty", field)
        normalized.append(text.lower() if lowercase else text)
    if len(set(normalized)) != len(normalized):
        code = "CFG_DUPLICATE_LABEL" if field == "label_space" else "CFG_DUPLICATE_ANNOTATOR"
        raise ConfigError(code, f"{field} contains duplicate values", field)
    return tuple(sorted(normalized))


def _parse_pair(
    raw: dict[str, Any], metric: str, annotators: tuple[str, ...]
) -> tuple[str, str] | None:
    if "pair" not in raw:
        return None
    pair_raw = raw["pair"]
    if metric != "cohen":
        raise ConfigError("CFG_PAIR_INVALID", "pair is only valid for cohen", "pair")
    if not isinstance(pair_raw, list) or len(pair_raw) != 2:
        raise ConfigError("CFG_PAIR_INVALID", "pair must contain exactly two ids", "pair")
    pair = tuple(_normal_text(item, "pair") for item in pair_raw)
    if pair[0] == pair[1]:
        raise ConfigError("CFG_PAIR_INVALID", "pair ids must be distinct", "pair")
    if annotators and any(item not in annotators for item in pair):
        raise ConfigError(
            "CFG_PAIR_INVALID", "pair ids must be in the annotator whitelist", "pair"
        )
    return pair  # type: ignore[return-value]


def _parse_dataset(raw: dict[str, Any], base: Path) -> DatasetSpec:
    dataset = _required(raw, "dataset", "dataset")
    if not isinstance(dataset, dict):
        raise ConfigError("CFG_TYPE_INVALID", "dataset must be a mapping", "dataset")
    unknown = sorted(set(dataset) - DATASET_FIELDS)
    if unknown:
        raise ConfigError(
            "CFG_UNKNOWN_FIELD", f"unknown dataset field {unknown[0]}", f"dataset.{unknown[0]}"
        )
    path_value = _normal_text(_required(dataset, "path", "dataset.path"), "dataset.path")
    file_format = _normal_text(
        _required(dataset, "format", "dataset.format"), "dataset.format"
    ).lower()
    if file_format not in FORMATS:
        raise ConfigError(
            "CFG_DATASET_FORMAT", f"unsupported dataset format {file_format}", "dataset.format"
        )
    encoding = str(dataset.get("encoding", "utf-8")).strip().lower()
    if encoding not in ENCODINGS:
        raise ConfigError(
            "CFG_ENCODING_UNKNOWN", f"unsupported encoding {encoding}", "dataset.encoding"
        )
    return DatasetSpec(_resolve(base, path_value), path_value, file_format, encoding)


def _parse_bool(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError("CFG_BOOL_INVALID", f"{key} must be a boolean", key)
    return value


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError("CFG_FILE_NOT_FOUND", f"config not found: {path}", "config") from exc
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError("CFG_PARSE_ERROR", f"cannot parse config: {exc}", "config") from exc
    if not isinstance(loaded, dict):
        raise ConfigError("CFG_TYPE_INVALID", "config root must be a mapping", "config")
    raw: dict[str, Any] = loaded
    unknown = sorted(set(raw) - TOP_LEVEL)
    if unknown:
        raise ConfigError("CFG_UNKNOWN_FIELD", f"unknown field {unknown[0]}", unknown[0])
    base = config_path.parent

    dataset = _parse_dataset(raw, base)
    labels = _string_list(_required(raw, "label_space", "label_space"), "label_space")
    annotators_raw = raw.get("annotators", [])
    if annotators_raw is None:
        annotators_raw = []
    if not isinstance(annotators_raw, list):
        raise ConfigError("CFG_TYPE_INVALID", "annotators must be a list", "annotators")
    annotators = (
        _string_list(annotators_raw, "annotators", lowercase=False)
        if annotators_raw else tuple()
    )
    metric = _normal_text(_required(raw, "metric", "metric"), "metric").lower()
    if metric not in METRICS:
        raise ConfigError("CFG_METRIC_UNKNOWN", f"unknown metric {metric}", "metric")
    pair = _parse_pair(raw, metric, annotators)
    min_annotators = _positive_int(raw, "min_annotators_per_item", 2)
    review_threshold = _threshold(raw, "review_threshold", 0.6)
    outlier_threshold = _threshold(raw, "outlier_threshold", 0.65)
    multi_label = _parse_bool(raw, "multi_label", False)
    strict = _parse_bool(raw, "strict", False)
    output_dir = _resolve(
        base, _normal_text(_required(raw, "output_dir", "output_dir"), "output_dir")
    )
    if raw.get("db_path") is not None:
        db_path = _resolve(base, _normal_text(raw["db_path"], "db_path"))
    else:
        db_path = (output_dir.parent / "var" / "tagconsensus.db").resolve()
    adjudication_file: Path | None = None
    if raw.get("adjudication_file") is not None:
        adjudication_file = _resolve(
            base, _normal_text(raw["adjudication_file"], "adjudication_file")
        )
    assignment_manifest: AssignmentManifestSpec | None = None
    if raw.get("assignment_manifest") is not None:
        assignment_manifest = parse_assignment_spec(raw["assignment_manifest"], base)

    semantic = {
        "dataset": {
            "path": str(dataset.path),
            "format": dataset.format,
            "encoding": dataset.encoding,
        },
        "label_space": list(labels),
        "annotators": list(annotators),
        "metric": metric,
        "pair": list(pair) if pair else None,
        "multi_label": multi_label,
        "min_annotators_per_item": min_annotators,
        "review_threshold": review_threshold,
        "outlier_threshold": outlier_threshold,
        "adjudication_file": str(adjudication_file) if adjudication_file else None,
        "assignment_manifest": _manifest_summary(assignment_manifest),
        "output_dir": str(output_dir),
        "db_path": str(db_path),
        "strict": strict,
    }
    return Config(
        config_path=config_path,
        dataset=dataset,
        label_space=labels,
        annotators=annotators,
        metric=metric,
        pair=pair,
        multi_label=multi_label,
        min_annotators_per_item=min_annotators,
        review_threshold=review_threshold,
        outlier_threshold=outlier_threshold,
        adjudication_file=adjudication_file,
        assignment_manifest=assignment_manifest,
        output_dir=output_dir,
        db_path=db_path,
        strict=strict,
        fingerprint=sha256_text(stable_json(semantic)),
    )
