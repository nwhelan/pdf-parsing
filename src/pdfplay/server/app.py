"""FastAPI backend for the visual comparison UI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import registry
from ..metrics import DOC_CLASSES, bank_statement, generic
from ..runner import run_parser
from ..workspace import Workspace

STATIC = Path(__file__).parent / "static"


def create_app(workspace: Workspace | None = None) -> FastAPI:
    ws = workspace or Workspace()
    app = FastAPI(title="pdfplay", version="0.1.0")

    # -- static (built React app) ----------------------------------------

    if (STATIC / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = STATIC / "index.html"
        if not page.exists():
            return (
                "<h1>pdfplay</h1><p>The front-end has not been built yet. Run:</p>"
                "<pre>cd web &amp;&amp; npm ci &amp;&amp; npm run build</pre>"
                "<p>The JSON API is up regardless — see <a href='/docs'>/docs</a>.</p>"
            )
        return page.read_text(encoding="utf-8")

    @app.get("/favicon.ico")
    def favicon() -> Response:
        return Response(status_code=204)

    # -- parsers ---------------------------------------------------------

    @app.get("/api/parsers")
    def parsers() -> list[dict[str, Any]]:
        return registry.describe_all()

    # -- presets ---------------------------------------------------------

    @app.get("/api/presets")
    def presets(parser_id: str = Query("")) -> list[dict[str, Any]]:
        return [p.as_dict() for p in ws.list_presets(parser_id)]

    @app.post("/api/presets")
    def save_preset(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            preset = ws.save_preset(
                name=str(payload.get("name") or ""),
                parser_id=str(payload.get("parser_id") or ""),
                options=payload.get("options") or {},
                notes=str(payload.get("notes") or ""),
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return preset.as_dict()

    @app.delete("/api/presets/{preset_id}")
    def delete_preset(preset_id: str) -> dict[str, bool]:
        ws.delete_preset(preset_id)
        return {"ok": True}

    # -- documents -------------------------------------------------------

    @app.get("/api/documents")
    def documents() -> list[dict[str, Any]]:
        return [m.as_dict() for m in ws.list_documents()]

    @app.post("/api/documents")
    async def upload(file: UploadFile, doc_class: str = Query("")) -> dict[str, Any]:
        data = await file.read()
        if not data.startswith(b"%PDF"):
            raise HTTPException(400, "not a PDF")
        meta = ws.add_bytes(data, file.filename or "upload.pdf", doc_class)
        return meta.as_dict()

    @app.get("/api/documents/{doc_id}")
    def document(doc_id: str) -> dict[str, Any]:
        try:
            meta = ws.get_document(doc_id)
        except KeyError:
            raise HTTPException(404, "unknown document") from None
        geo = ws.geometry(doc_id)
        return {
            **meta.as_dict(),
            "geometry": [
                {"page": g.number, "width": g.width, "height": g.height, "rotation": g.rotation}
                for g in geo.values()
            ],
            "results": ws.list_results(doc_id),
            "has_ground_truth": ws.get_ground_truth(doc_id) is not None,
        }

    @app.delete("/api/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict[str, bool]:
        ws.delete_document(doc_id)
        return {"ok": True}

    @app.get("/api/documents/{doc_id}/pages/{page}/image")
    def page_image(doc_id: str, page: int, scale: float = Query(2.0, ge=0.25, le=6.0)) -> FileResponse:
        try:
            path = ws.render(doc_id, page, scale)
        except (KeyError, IndexError):
            raise HTTPException(404, "unknown page") from None
        return FileResponse(path, media_type="image/png")

    @app.put("/api/documents/{doc_id}/ground-truth")
    def set_ground_truth(doc_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, bool]:
        ws.set_ground_truth(doc_id, payload)
        return {"ok": True}

    @app.get("/api/documents/{doc_id}/ground-truth")
    def get_ground_truth(doc_id: str) -> Any:
        return ws.get_ground_truth(doc_id) or {}

    # -- runs ------------------------------------------------------------

    @app.post("/api/documents/{doc_id}/parse/{parser_id}")
    def parse(doc_id: str, parser_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        pages = payload.get("pages")
        options = payload.get("options") or {}
        force = bool(payload.get("force"))
        try:
            result, key, cached = run_parser(ws, doc_id, parser_id, pages, options, force)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from None
        return {"key": key, "cached": cached, "result": json.loads(result.model_dump_json())}

    @app.get("/api/documents/{doc_id}/results/{key}")
    def result(doc_id: str, key: str) -> Any:
        found = ws.load_result(doc_id, key)
        if found is None:
            raise HTTPException(404, "no such result")
        return JSONResponse(json.loads(found.model_dump_json()))

    @app.delete("/api/documents/{doc_id}/results/{key}")
    def drop_result(doc_id: str, key: str) -> dict[str, bool]:
        ws.delete_result(doc_id, key)
        return {"ok": True}

    # -- scoring ---------------------------------------------------------

    @app.post("/api/documents/{doc_id}/score")
    def score(doc_id: str, payload: dict[str, Any] = Body(default={})) -> dict[str, Any]:
        keys: list[str] = payload.get("keys") or [r["key"] for r in ws.list_results(doc_id)]
        doc_class = payload.get("doc_class") or ws.get_document(doc_id).doc_class
        loaded = [(k, ws.load_result(doc_id, k)) for k in keys]
        pairs = [(k, r) for k, r in loaded if r is not None]
        results = [r for _, r in pairs]

        rows = []
        truth = ws.get_ground_truth(doc_id) or {}
        ledger = truth.get("transactions") if isinstance(truth, dict) else None

        for key, result in pairs:
            row: dict[str, Any] = {"key": key}
            row.update(generic.analyze(result))
            if doc_class == "bank_statement":
                report = bank_statement.analyze(result)
                row["bank_statement"] = report.as_dict()
                if ledger:
                    row["ledger_score"] = bank_statement.score_against_ledger(result, ledger)
            rows.append(row)

        return {
            "doc_class": doc_class,
            "known_classes": sorted(DOC_CLASSES),
            "rows": rows,
            "similarity": generic.similarity_matrix(results),
        }

    @app.post("/api/documents/{doc_id}/diff")
    def diff(doc_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        left = ws.load_result(doc_id, payload["left"])
        right = ws.load_result(doc_id, payload["right"])
        if left is None or right is None:
            raise HTTPException(404, "missing result")
        return generic.compare(left, right)

    return app


app = create_app()
