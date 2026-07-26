from __future__ import annotations

import json

import pytest

from pdfplay.runner import run_parser
from pdfplay.workspace import Workspace


def test_adding_the_same_pdf_twice_is_idempotent(workspace: Workspace, borderless):
    first = workspace.add_document(borderless.path, doc_class="bank_statement")
    second = workspace.add_document(borderless.path)
    assert first.doc_id == second.doc_id
    assert len(workspace.list_documents()) == 1
    assert second.doc_class == "bank_statement"


def test_results_are_cached_until_forced(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    _, key, cached = run_parser(workspace, meta.doc_id, "pymupdf")
    assert cached is False
    _, same_key, cached_again = run_parser(workspace, meta.doc_id, "pymupdf")
    assert (same_key, cached_again) == (key, True)
    _, _, forced = run_parser(workspace, meta.doc_id, "pymupdf", force=True)
    assert forced is False


def test_different_options_get_different_cache_keys(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    _, key_a, _ = run_parser(workspace, meta.doc_id, "pdfplumber", options={"table_strategy": "lines"})
    _, key_b, _ = run_parser(workspace, meta.doc_id, "pdfplumber", options={"table_strategy": "text"})
    assert key_a != key_b
    assert len(workspace.list_results(meta.doc_id)) == 2


def test_a_broken_parser_is_captured_not_raised(workspace: Workspace, borderless, monkeypatch):
    from pdfplay import registry

    cls = registry.get("pymupdf")
    monkeypatch.setattr(cls, "parse", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    meta = workspace.add_document(borderless.path)
    result, _, _ = run_parser(workspace, meta.doc_id, "pymupdf", force=True)
    assert result.status == "error"
    assert "boom" in result.error


def test_unknown_parser_raises(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    with pytest.raises(KeyError):
        run_parser(workspace, meta.doc_id, "does-not-exist")


def test_page_render_is_cached_on_disk(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    path = workspace.render(meta.doc_id, 1, scale=1.0)
    assert path.exists() and path.stat().st_size > 1000
    assert workspace.render(meta.doc_id, 1, scale=1.0) == path


# -- API --------------------------------------------------------------------


@pytest.fixture
def client(workspace: Workspace, borderless):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from pdfplay.server.app import create_app

    workspace.add_document(borderless.path, doc_class="bank_statement")
    workspace.set_ground_truth(
        workspace.list_documents()[0].doc_id, {"transactions": borderless.ledger["transactions"]}
    )
    return fastapi_testclient.TestClient(create_app(workspace))


def test_api_lists_parsers_and_documents(client):
    parsers = client.get("/api/parsers").json()
    assert any(p["id"] == "pymupdf" and p["available"] for p in parsers)
    docs = client.get("/api/documents").json()
    assert len(docs) == 1


def test_one_broken_adapter_does_not_empty_the_parser_list(client, monkeypatch):
    """The sidebar is built from this list, so it has to survive a bad probe.

    Availability probes import packages and shell out to binaries, so on
    someone else's machine any of them can raise. Before, that 500'd the whole
    endpoint and the viewer could only say "failed to fetch".
    """
    from pdfplay import registry

    def explode():
        raise OSError("tesseract is not installed or it's not in your PATH")

    monkeypatch.setattr(registry.get("tesseract"), "check_availability", staticmethod(explode))

    response = client.get("/api/parsers")
    assert response.status_code == 200

    parsers = {p["id"]: p for p in response.json()}
    assert len(parsers) == len(registry.all_parsers()), "every parser is still listed"
    assert parsers["pymupdf"]["available"] is True, "a healthy parser is unaffected"
    assert parsers["tesseract"]["available"] is False
    assert "not in your PATH" in parsers["tesseract"]["unavailable_reason"], "the reason is reported"


def test_a_broken_probe_does_not_break_run_all(monkeypatch):
    from pdfplay import registry

    def explode():
        raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(registry.get("tesseract"), "check_availability", staticmethod(explode))
    ids = [c.id for c in registry.available_parsers()]
    assert "pymupdf" in ids
    assert "tesseract" not in ids


def test_api_parse_score_and_diff(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]

    first = client.post(f"/api/documents/{doc_id}/parse/pymupdf", json={}).json()
    assert first["result"]["status"] == "ok"
    assert first["cached"] is False
    assert client.post(f"/api/documents/{doc_id}/parse/pymupdf", json={}).json()["cached"] is True

    second = client.post(f"/api/documents/{doc_id}/parse/pdfplumber", json={}).json()

    scores = client.post(f"/api/documents/{doc_id}/score", json={}).json()
    assert scores["doc_class"] == "bank_statement"
    assert {r["parser_id"] for r in scores["rows"]} == {"pymupdf", "pdfplumber"}
    for row in scores["rows"]:
        assert row["bank_statement"]["reconciliation_rate"] == 1.0
        assert row["ledger_score"]["f1"] == 1.0

    diff = client.post(
        f"/api/documents/{doc_id}/diff", json={"left": first["key"], "right": second["key"]}
    ).json()
    assert 0.0 <= diff["text_similarity"] <= 1.0
    assert 0.0 <= diff["line_similarity"] <= 1.0


def test_api_serves_a_page_image(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    res = client.get(f"/api/documents/{doc_id}/pages/1/image", params={"scale": 1})
    assert res.status_code == 200
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_rejects_non_pdf_upload(client):
    res = client.post("/api/documents", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert res.status_code == 400


def test_api_serves_the_built_front_end(client):
    """`pdfplay serve` must work straight from a pip install — no node step."""
    from pdfplay.server.app import STATIC

    res = client.get("/")
    assert res.status_code == 200
    if (STATIC / "index.html").exists():
        assert 'id="root"' in res.text
        asset = next((STATIC / "assets").glob("*.js"), None)
        assert asset is not None, "built JS bundle missing"
        assert client.get(f"/assets/{asset.name}").status_code == 200
    else:  # front-end not built in this checkout
        assert "npm run build" in res.text


def test_api_reports_unknown_document(client):
    assert client.get("/api/documents/deadbeef").status_code == 404


def test_ground_truth_round_trips(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    payload = {"transactions": [{"date": "03/01/2025", "amount": -1.0, "balance": 9.0}]}
    assert client.put(f"/api/documents/{doc_id}/ground-truth", json=payload).status_code == 200
    assert client.get(f"/api/documents/{doc_id}/ground-truth").json() == json.loads(json.dumps(payload))
