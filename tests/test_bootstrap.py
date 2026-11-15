from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from tagconsensus.cli import run_demo
from tagconsensus.config import load_config
from tagconsensus.errors import ConfigError, MetricError
from tagconsensus.ledger import get_run, latest_run
from tagconsensus.metrics import bootstrap_interval, compute_metric
from tagconsensus.models import resample_dataset

from conftest import make_config, make_dataset

BOOTSTRAP_DEFAULTS = {
    "replicates": 1000,
    "confidence": 0.95,
    "seed": 20240521,
    "min_valid_ratio": 0.8,
}


def _config(tmp_path, *, metric="krippendorff_nominal", bootstrap=None, **kwargs):
    extra = dict(kwargs.pop("extra", None) or {})
    if bootstrap is not None:
        extra["bootstrap"] = bootstrap
    return load_config(make_config(tmp_path, metric=metric, extra=extra, **kwargs))


def _mixed_dataset():
    """it_1 has two disagreeing annotations; it_2/it_3 have one each.

    A bootstrap draw of three clusters is degenerate (METRIC_NO_ITEMS) exactly
    when it contains no it_1 occurrence; a valid draw with k occurrences of
    it_1 yields alpha = (1 - k) / k, i.e. 0, -1/2 or -2/3.
    """
    return make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_alpha"),
        ("A02", "it_3", "topic_beta"),
    ])


def _oracle_draw_value(dataset, draw):
    """Route-B reference oracle: multiplicity-weighted sufficient statistics.

    Each distinct item cluster contributes its per-occurrence statistics
    weighted by how many times the draw sampled it. Returns the MetricError
    code string for degenerate draws.
    """
    clusters: dict[str, Counter] = {}
    for row in dataset.annotations:
        if row.is_missing:
            continue
        clusters.setdefault(row.item_id, Counter())[row.labels[0]] += 1
    multiplicities = Counter(draw)
    eligible = [
        (clusters[item_id], count)
        for item_id, count in multiplicities.items()
        if sum(clusters[item_id].values()) >= 2
    ]
    if not eligible:
        return "METRIC_NO_ITEMS"
    total_n = sum(count * sum(counter.values()) for counter, count in eligible)
    diagonal = 0.0
    marginals: Counter = Counter()
    for counter, count in eligible:
        size = sum(counter.values())
        for label, value in counter.items():
            diagonal += count * value * (value - 1) / (size - 1)
            marginals[label] += count * value
    do = 1.0 - diagonal / total_n
    de = (total_n * total_n - sum(value * value for value in marginals.values())) / (
        total_n * (total_n - 1)
    )
    if de <= 1e-12:
        return "METRIC_PE_ONE"
    return 1.0 - do / de


def _reference_percentile(sorted_values, q):
    count = len(sorted_values)
    if count == 1:
        return sorted_values[0]
    rank = q * (count - 1)
    low = int(rank)
    high = min(low + 1, count - 1)
    fraction = rank - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def test_resample_dataset_keeps_duplicate_clusters_independent(tmp_path):
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_beta"),
        ("A01", "it_2", "topic_alpha"),
        ("A02", "it_2", "topic_alpha"),
    ])
    resampled = resample_dataset(dataset, ["it_1", "it_1"])
    # The duplicated cluster survives as two occurrences: item_id dedup and
    # per-item bucketing must not merge them back into one unit.
    assert len(resampled.item_ids) == 2
    assert resampled.records == 4
    synthetic = resampled.item_ids
    assert synthetic[0] != synthetic[1]
    for item_id in synthetic:
        rows = [row for row in resampled.annotations if row.item_id == item_id]
        assert sorted(row.annotator_id for row in rows) == ["A01", "A02"]
    # The source dataset is immutable.
    assert dataset.records == 4
    assert dataset.item_ids == ("it_1", "it_2")
    config = _config(tmp_path, bootstrap={"replicates": 0})
    value = compute_metric(resampled, config, bootstrap=False).value
    # Two independent {alpha, beta} occurrences give alpha = -0.5; merging
    # them into one {alpha: 2, beta: 2} bucket would wrongly give 0.0.
    assert value == pytest.approx(-0.5, abs=1e-12)


def test_route_a_matches_multiplicity_oracle_for_every_draw(tmp_path):
    dataset = _mixed_dataset()
    replicates, seed = 200, 97
    config = _config(
        tmp_path,
        bootstrap={"replicates": replicates, "seed": seed, "min_valid_ratio": 0.0},
    )
    result = bootstrap_interval(dataset, config)
    assert result is not None
    assert result.status == "ok"
    item_ids = list(dataset.item_ids)
    rng = random.Random(seed)
    oracle_values = []
    oracle_degenerate = 0
    for _ in range(replicates):
        draw = [item_ids[rng.randrange(len(item_ids))] for _ in item_ids]
        resampled = resample_dataset(dataset, draw)
        try:
            route_a = compute_metric(resampled, config, bootstrap=False).value
        except MetricError as exc:
            route_a = exc.code
        oracle = _oracle_draw_value(dataset, draw)
        if isinstance(oracle, str):
            assert route_a == oracle
            oracle_degenerate += 1
        else:
            assert route_a == pytest.approx(oracle, abs=1e-12)
            oracle_values.append(oracle)
    assert oracle_degenerate > 0  # the fixed seed really hits degenerate draws
    assert result.requested == replicates
    assert result.valid == len(oracle_values)
    assert result.degenerate == oracle_degenerate
    assert result.degenerate_codes == (("METRIC_NO_ITEMS", oracle_degenerate),)
    oracle_values.sort()
    assert result.lower == pytest.approx(
        _reference_percentile(oracle_values, 0.025), abs=1e-12
    )
    assert result.upper == pytest.approx(
        _reference_percentile(oracle_values, 0.975), abs=1e-12
    )


def test_degenerate_de_zero_draws_are_counted_not_zeroed(tmp_path):
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
        ("A01", "it_2", "topic_beta"),
        ("A02", "it_2", "topic_beta"),
    ])
    config = _config(
        tmp_path,
        bootstrap={"replicates": 200, "seed": 11, "min_valid_ratio": 0.0},
    )
    result = bootstrap_interval(dataset, config)
    assert result.degenerate > 0
    assert result.degenerate_codes == (("METRIC_PE_ONE", result.degenerate),)
    assert result.valid + result.degenerate == result.requested == 200
    # Every valid draw has perfect agreement; treating degenerate draws as
    # alpha = 0 would drag the lower bound away from 1.0.
    assert result.lower == pytest.approx(1.0)
    assert result.upper == pytest.approx(1.0)


def test_min_valid_ratio_controls_availability(tmp_path):
    dataset = _mixed_dataset()
    base = {"replicates": 200, "seed": 5}
    ok = bootstrap_interval(
        dataset, _config(tmp_path, bootstrap={**base, "min_valid_ratio": 0.0})
    )
    assert ok.status == "ok"
    ratio = ok.valid / ok.requested
    assert 0.0 < ratio < 1.0
    rejected = bootstrap_interval(
        dataset, _config(tmp_path, bootstrap={**base, "min_valid_ratio": ratio + 0.01})
    )
    assert rejected.status == "unavailable"
    assert rejected.unavailable_reason == "INSUFFICIENT_VALID_REPLICATES"
    assert rejected.lower is None
    assert rejected.upper is None
    # Same seed -> same draws; the requested count is never topped up or
    # silently reduced when draws degenerate.
    assert (rejected.requested, rejected.valid, rejected.degenerate) == (
        ok.requested,
        ok.valid,
        ok.degenerate,
    )
    boundary = bootstrap_interval(
        dataset, _config(tmp_path, bootstrap={**base, "min_valid_ratio": ratio})
    )
    assert boundary.status == "ok"  # the threshold comparison is strict


def test_no_valid_replicates_is_unavailable(tmp_path):
    dataset = make_dataset([
        ("A01", "it_1", "topic_alpha"),
        ("A02", "it_1", "topic_alpha"),
    ])
    config = _config(
        tmp_path,
        bootstrap={"replicates": 50, "seed": 3, "min_valid_ratio": 0.0},
    )
    result = bootstrap_interval(dataset, config)
    assert result.status == "unavailable"
    assert result.unavailable_reason == "NO_VALID_REPLICATES"
    assert result.valid == 0
    assert result.degenerate == 50
    assert result.degenerate_codes == (("METRIC_PE_ONE", 50),)
    assert result.lower is None
    assert result.upper is None


def test_fixed_seed_replays_exactly(tmp_path):
    dataset = _mixed_dataset()
    config = _config(tmp_path, bootstrap={"replicates": 150, "seed": 424242})
    first = bootstrap_interval(dataset, config)
    second = bootstrap_interval(dataset, config)
    assert first == second
    assert compute_metric(dataset, config).bootstrap == first


def test_different_seed_changes_the_draw_sequence(tmp_path):
    dataset = _mixed_dataset()
    first = bootstrap_interval(
        dataset, _config(tmp_path, bootstrap={"replicates": 150, "seed": 1})
    )
    second = bootstrap_interval(
        dataset, _config(tmp_path, bootstrap={"replicates": 150, "seed": 2})
    )
    assert (first.valid, first.lower, first.upper) != (
        second.valid,
        second.lower,
        second.upper,
    )


def test_point_estimate_unchanged_and_bootstrap_attached(tmp_path):
    dataset = _mixed_dataset()
    config = _config(tmp_path, bootstrap={"replicates": 100, "seed": 1})
    with_bootstrap = compute_metric(dataset, config)
    without = compute_metric(dataset, config, bootstrap=False)
    assert with_bootstrap.bootstrap is not None
    assert without.bootstrap is None
    for field in (
        "value",
        "po",
        "pe",
        "do",
        "de",
        "eligible_items",
        "eligible_annotations",
    ):
        assert getattr(with_bootstrap, field) == getattr(without, field)


def test_bootstrap_only_applies_to_krippendorff(tmp_path):
    dataset = _mixed_dataset()
    config = _config(
        tmp_path, metric="pairwise_agreement", bootstrap={"replicates": 100}
    )
    assert compute_metric(dataset, config).bootstrap is None


def test_zero_replicates_disables_bootstrap(tmp_path):
    dataset = _mixed_dataset()
    config = _config(tmp_path, bootstrap={"replicates": 0})
    assert compute_metric(dataset, config).bootstrap is None


def test_multi_label_bootstrap_tracks_macro_average(tmp_path):
    rows = [
        ("A01", "it_1", ["topic_alpha", "topic_beta"]),
        ("A02", "it_1", ["topic_alpha"]),
        ("A03", "it_1", ["topic_alpha", "topic_beta"]),
        ("A01", "it_2", ["topic_beta"]),
        ("A02", "it_2", ["topic_beta"]),
        ("A03", "it_2", ["topic_alpha"]),
        ("A01", "it_3", ["topic_beta"]),
        ("A02", "it_3", ["topic_beta"]),
        ("A03", "it_3", ["topic_beta"]),
    ]
    config = _config(
        tmp_path,
        multi_label=True,
        bootstrap={"replicates": 50, "seed": 8},
    )
    result = compute_metric(make_dataset(rows), config)
    assert result.bootstrap is not None
    assert result.bootstrap.requested == 50
    assert result.bootstrap.valid + result.bootstrap.degenerate == 50
    assert result.bootstrap.status == "ok"
    assert result.bootstrap.lower <= result.bootstrap.upper


def test_bootstrap_config_defaults(tmp_path):
    config = _config(tmp_path)
    assert config.bootstrap.replicates == BOOTSTRAP_DEFAULTS["replicates"]
    assert config.bootstrap.confidence == BOOTSTRAP_DEFAULTS["confidence"]
    assert config.bootstrap.seed == BOOTSTRAP_DEFAULTS["seed"]
    assert config.bootstrap.min_valid_ratio == BOOTSTRAP_DEFAULTS["min_valid_ratio"]
    summary = config.summary()["bootstrap"]
    assert summary == BOOTSTRAP_DEFAULTS


@pytest.mark.parametrize(
    "payload,code,field",
    [
        ({"replicates": -1}, "CFG_INT_RANGE", "bootstrap.replicates"),
        ({"replicates": 2.5}, "CFG_INT_RANGE", "bootstrap.replicates"),
        ({"replicates": True}, "CFG_INT_RANGE", "bootstrap.replicates"),
        ({"seed": "abc"}, "CFG_INT_RANGE", "bootstrap.seed"),
        ({"seed": -3}, "CFG_INT_RANGE", "bootstrap.seed"),
        ({"confidence": 1.5}, "CFG_THRESHOLD_RANGE", "bootstrap.confidence"),
        ({"confidence": 0}, "CFG_THRESHOLD_RANGE", "bootstrap.confidence"),
        ({"confidence": "high"}, "CFG_THRESHOLD_RANGE", "bootstrap.confidence"),
        ({"min_valid_ratio": 1.5}, "CFG_THRESHOLD_RANGE", "bootstrap.min_valid_ratio"),
        ({"min_valid_ratio": -0.1}, "CFG_THRESHOLD_RANGE", "bootstrap.min_valid_ratio"),
        ({"unknown": 1}, "CFG_UNKNOWN_FIELD", "bootstrap.unknown"),
    ],
)
def test_bootstrap_config_validation(tmp_path, payload, code, field):
    with pytest.raises(ConfigError) as error:
        _config(tmp_path, bootstrap=payload)
    assert error.value.code == code
    assert error.value.field == field


def test_bootstrap_section_must_be_a_mapping(tmp_path):
    config_path = make_config(
        tmp_path, metric="krippendorff_nominal", extra={"bootstrap": 5}
    )
    with pytest.raises(ConfigError) as error:
        load_config(config_path)
    assert error.value.code == "CFG_TYPE_INVALID"
    assert error.value.field == "bootstrap"


def test_bootstrap_parameters_enter_config_fingerprint(tmp_path):
    base = _config(tmp_path)
    other = _config(tmp_path, bootstrap={"seed": 99})
    assert base.fingerprint != other.fingerprint


def _krippendorff_pipeline(tmp_path, bootstrap):
    data = tmp_path / "data.csv"
    data.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\n"
        "A01,it_3,topic_alpha\nA02,it_3,topic_beta\n",
        encoding="utf-8",
    )
    return make_config(
        tmp_path,
        data,
        metric="krippendorff_nominal",
        annotators=["A01", "A02"],
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
        extra={"bootstrap": bootstrap},
    )


def test_report_and_ledger_carry_bootstrap(tmp_path):
    config_path = _krippendorff_pipeline(
        tmp_path,
        {"replicates": 60, "seed": 17, "confidence": 0.9, "min_valid_ratio": 0.5},
    )
    assert run_demo(str(config_path)) == 0
    report = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    bootstrap = report["metrics"]["bootstrap"]
    assert bootstrap["method"] == "item_cluster_percentile"
    assert bootstrap["seed"] == 17
    assert bootstrap["confidence"] == 0.9
    assert bootstrap["requested"] == 60
    assert bootstrap["valid"] + bootstrap["degenerate"] == 60
    assert bootstrap["status"] == "ok"
    assert bootstrap["lower"] <= bootstrap["upper"]
    markdown = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
    assert "Bootstrap 区间" in markdown
    assert "`17`" in markdown
    html = (tmp_path / "output" / "report.html").read_text(encoding="utf-8")
    assert "Bootstrap" in html
    detail = get_run(
        tmp_path / "var" / "tagconsensus.db",
        latest_run(tmp_path / "var" / "tagconsensus.db")["run_id"],
    )
    rows = detail["bootstrap_intervals"]
    assert len(rows) == 1
    row = rows[0]
    assert row["seed"] == 17
    assert row["requested"] == 60
    assert row["valid"] == bootstrap["valid"]
    assert row["degenerate"] == bootstrap["degenerate"]
    assert row["status"] == "ok"
    assert row["lower"] == pytest.approx(bootstrap["lower"])
    assert row["upper"] == pytest.approx(bootstrap["upper"])
    # Replaying the same config and seed reproduces the identical interval.
    assert run_demo(str(config_path)) == 0
    replayed = json.loads(
        (tmp_path / "output" / "report.json").read_text(encoding="utf-8")
    )
    assert replayed["metrics"]["bootstrap"] == bootstrap


def test_unavailable_bootstrap_flows_to_report_and_ledger(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_beta\n"
        "A01,it_2,topic_alpha\n"
        "A02,it_3,topic_beta\n",
        encoding="utf-8",
    )
    config_path = make_config(
        tmp_path,
        data,
        metric="krippendorff_nominal",
        annotators=["A01", "A02"],
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
        extra={"bootstrap": {"replicates": 60, "seed": 23, "min_valid_ratio": 1.0}},
    )
    assert run_demo(str(config_path)) == 0
    report = json.loads((tmp_path / "output" / "report.json").read_text(encoding="utf-8"))
    bootstrap = report["metrics"]["bootstrap"]
    assert bootstrap["status"] == "unavailable"
    assert bootstrap["unavailable_reason"] == "INSUFFICIENT_VALID_REPLICATES"
    assert bootstrap["requested"] == 60
    assert bootstrap["degenerate"] > 0
    assert bootstrap["degenerate_codes"] == {"METRIC_NO_ITEMS": bootstrap["degenerate"]}
    assert bootstrap["lower"] is None
    assert bootstrap["upper"] is None
    # The point estimate and conclusion are unaffected by the unavailable interval.
    assert report["metrics"]["value"] == pytest.approx(0.0, abs=1e-12)
    markdown = (tmp_path / "output" / "report.md").read_text(encoding="utf-8")
    assert "不可用" in markdown
    detail = get_run(
        tmp_path / "var" / "tagconsensus.db",
        latest_run(tmp_path / "var" / "tagconsensus.db")["run_id"],
    )
    row = detail["bootstrap_intervals"][0]
    assert row["status"] == "unavailable"
    assert row["unavailable_reason"] == "INSUFFICIENT_VALID_REPLICATES"
    assert row["lower"] is None
    assert row["upper"] is None
    assert row["valid"] == bootstrap["valid"]
    assert json.loads(row["degenerate_codes"]) == bootstrap["degenerate_codes"]
