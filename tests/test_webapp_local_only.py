from __future__ import annotations

from pathlib import Path

from tagconsensus.cli import run_demo
from tagconsensus.ledger import latest_run
from tagconsensus.webapp import create_app

from conftest import PROJECT_ROOT, make_config


def _run_fixture(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text(
        "annotator_id,item_id,label\n"
        "A01,it_1,topic_alpha\nA02,it_1,topic_alpha\nA03,it_1,topic_beta\n"
        "A01,it_2,topic_beta\nA02,it_2,topic_beta\nA03,it_2,topic_beta\n",
        encoding="utf-8",
    )
    config_path = make_config(
        tmp_path, data, metric="pairwise_agreement",
        annotators=["A01", "A02", "A03"], review_threshold=0.8,
        output_dir=tmp_path / "output",
        db_path=tmp_path / "var" / "tagconsensus.db",
    )
    assert run_demo(str(config_path)) == 0
    return tmp_path / "var" / "tagconsensus.db"


def _client(db_path):
    app = create_app(str(db_path))
    app.config.update(TESTING=True)
    return app.test_client()


def test_empty_state_pages_return_200(tmp_path):
    client = _client(tmp_path / "empty.db")
    for route in ("/", "/review", "/ledger"):
        response = client.get(route)
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "尚无运行记录" in client.get("/").get_data(as_text=True)
    assert "尚无运行记录" in client.get("/review").get_data(as_text=True)
    assert "尚无历史运行" in client.get("/ledger").get_data(as_text=True)


def test_three_pages_render_with_run(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    overview = client.get("/")
    assert overview.status_code == 200
    assert "需复核" in overview.get_data(as_text=True)
    review = client.get("/review")
    assert review.status_code == 200
    assert "LOW_CONSENSUS" in review.get_data(as_text=True)
    assert "it_1" in review.get_data(as_text=True)
    ledger = client.get("/ledger")
    assert ledger.status_code == 200
    run_id = latest_run(db_path)["run_id"]
    assert run_id in ledger.get_data(as_text=True)
    detail = client.get(f"/ledger?run_id={run_id}")
    assert detail.status_code == 200
    assert "单次明细" in detail.get_data(as_text=True)
    assert "report.json" in detail.get_data(as_text=True)


def test_review_reason_filter_is_whitelisted(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    low = client.get("/review?reason=LOW_CONSENSUS")
    assert low.status_code == 200
    assert "it_1" in low.get_data(as_text=True)
    tie = client.get("/review?reason=TIE")
    assert tie.status_code == 200
    assert "it_1" not in tie.get_data(as_text=True)
    invalid = client.get("/review?reason=WHATEVER")
    assert invalid.status_code == 400
    assert invalid.headers["X-Content-Type-Options"] == "nosniff"


def test_unknown_query_parameters_are_rejected(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    assert client.get("/?path=/etc/passwd").status_code == 400
    assert client.get("/ledger?file=attachment.txt").status_code == 400
    run_id = latest_run(db_path)["run_id"]
    assert client.get(f"/ledger?run_id={run_id}&extra=1").status_code == 400


def test_invalid_and_unknown_run_ids(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    assert client.get("/ledger?run_id=bad%20id").status_code == 400
    assert client.get("/ledger?run_id=" + "0" * 32).status_code == 404


def test_pages_do_not_reference_external_assets_or_listen_publicly(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    scheme = "http" + "://"
    for route in ("/", "/review", "/ledger"):
        body = client.get(route).get_data(as_text=True)
        assert scheme not in body
        assert "https" + "://" not in body
        assert "cdn" not in body.lower()
    source = (PROJECT_ROOT / "src" / "tagconsensus" / "webapp.py").read_text(
        encoding="utf-8"
    )
    templates = (PROJECT_ROOT / "src" / "tagconsensus" / "templates")
    for path in templates.glob("*.html"):
        source += path.read_text(encoding="utf-8")
    forbidden_host = ".".join(["0", "0", "0", "0"])
    assert forbidden_host not in source
    assert scheme not in source


def test_read_only_routes_reject_writes(tmp_path):
    db_path = _run_fixture(tmp_path)
    client = _client(db_path)
    assert client.post("/").status_code == 405
    assert client.post("/review").status_code == 405
    assert client.post("/ledger").status_code == 405
    assert client.get("/review").headers["X-Frame-Options"] == "DENY"


def test_no_network_imports_in_web_surface():
    source = "\n".join(
        (PROJECT_ROOT / "src" / "tagconsensus" / name).read_text(encoding="utf-8")
        for name in ("webapp.py", "cli.py")
    )
    forbidden = ["requests", "urllib", "socket", "httpx"]
    assert not any(name in source for name in forbidden)
