from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tagconsensus.config import load_config
from tagconsensus.errors import ConfigError

from conftest import PROJECT_ROOT, make_config


def _mutate(path: Path, mutator) -> Path:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutator(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_demo_config_values_and_paths():
    config = load_config(PROJECT_ROOT / "configs" / "demo.yaml")
    assert config.metric == "krippendorff_nominal"
    assert config.label_space == tuple(sorted(config.label_space))
    assert config.annotators == ("A01", "A02", "A03", "A04", "A05", "A06")
    assert config.min_annotators_per_item == 3
    assert config.dataset.path == PROJECT_ROOT / "data" / "raw" / "annotations_demo.csv"
    assert config.output_dir == PROJECT_ROOT / "output"
    assert config.db_path == PROJECT_ROOT / "var" / "tagconsensus.db"
    assert config.strict is False


def test_strict_config_is_explicit():
    config = load_config(PROJECT_ROOT / "configs" / "strict.yaml")
    assert config.strict is True
    assert config.output_dir.name == "output_strict"


def test_fingerprint_is_stable(tmp_path):
    path = make_config(tmp_path)
    first = load_config(path)
    second = load_config(path)
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64


def test_missing_required_field_has_path(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.pop("metric"))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_MISSING_FIELD"
    assert error.value.field == "metric"


def test_unknown_metric(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.update(metric="kappa9"))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_METRIC_UNKNOWN"
    assert error.value.field == "metric"


@pytest.mark.parametrize("field", ["review_threshold", "outlier_threshold"])
def test_threshold_range(tmp_path, field):
    path = _mutate(make_config(tmp_path), lambda data: data.update({field: 1.2}))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_THRESHOLD_RANGE"
    assert error.value.field == field


def test_threshold_rejects_boolean(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.update(review_threshold=True))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_THRESHOLD_RANGE"


def test_empty_label_is_rejected(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data["label_space"].append("  "))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_LABEL_EMPTY"
    assert error.value.field == "label_space"
    path = _mutate(make_config(tmp_path), lambda data: data.update(label_space=[]))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_LABEL_EMPTY"


def test_duplicate_label_is_rejected(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data["label_space"].append("OTHER"))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_DUPLICATE_LABEL"


def test_unknown_field_is_rejected(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.update(magic=True))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_UNKNOWN_FIELD"
    assert error.value.field == "magic"


def test_dataset_format_and_encoding_are_checked(tmp_path):
    format_path = _mutate(
        make_config(tmp_path), lambda data: data["dataset"].update(format="parquet")
    )
    with pytest.raises(ConfigError) as error:
        load_config(format_path)
    assert error.value.code == "CFG_DATASET_FORMAT"
    encoding_path = _mutate(
        make_config(tmp_path), lambda data: data["dataset"].update(encoding="latin-1")
    )
    with pytest.raises(ConfigError) as error:
        load_config(encoding_path)
    assert error.value.code == "CFG_ENCODING_UNKNOWN"


def test_pair_only_for_cohen_and_must_be_whitelisted(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.update(pair=["A01", "A02"]))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_PAIR_INVALID"
    path = make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02"],
        extra={"pair": ["A01", "A09"]},
    )
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_PAIR_INVALID"


def test_pair_is_kept_when_valid(tmp_path):
    path = make_config(
        tmp_path, metric="cohen", annotators=["A01", "A02"],
        extra={"pair": ["A01", "A02"]},
    )
    config = load_config(path)
    assert config.pair == ("A01", "A02")


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError) as error:
        load_config(tmp_path / "nope.yaml")
    assert error.value.code == "CFG_FILE_NOT_FOUND"


def test_bad_boolean_and_min_annotators(tmp_path):
    path = _mutate(make_config(tmp_path), lambda data: data.update(multi_label="yes"))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_BOOL_INVALID"
    path = _mutate(make_config(tmp_path), lambda data: data.update(min_annotators_per_item=1))
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert error.value.code == "CFG_INT_RANGE"
