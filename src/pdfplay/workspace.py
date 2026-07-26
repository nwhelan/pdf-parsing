"""On-disk workspace: documents, cached page renders, and parse results.

Layout::

    workspace/
      docs/<doc_id>/source.pdf
      docs/<doc_id>/meta.json
      docs/<doc_id>/renders/p3@2.0.png
      docs/<doc_id>/results/pymupdf__a1b2c3d4.json
      docs/<doc_id>/ground_truth.json

``doc_id`` is the first 16 hex chars of the file's SHA-256, so re-adding the
same PDF is idempotent and results stay attached to content, not filenames.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .geometry import page_count, page_geometry, render_page
from .models import ParseResult

DEFAULT_WORKSPACE = Path(os.environ.get("PDFPLAY_WORKSPACE", "workspace"))


@dataclass
class DocumentMeta:
    doc_id: str
    name: str
    pages: int
    size_bytes: int
    sha256: str
    added_at: str
    doc_class: str = ""
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def options_key(parser_id: str, options: dict[str, Any] | None, pages: list[int] | None) -> str:
    payload = json.dumps(
        {"options": options or {}, "pages": sorted(pages) if pages else None},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:8]
    return f"{parser_id}__{digest}"


class Workspace:
    def __init__(self, root: Path | str = DEFAULT_WORKSPACE) -> None:
        self.root = Path(root)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)

    # -- documents -------------------------------------------------------

    def add_document(self, source: Path | str, name: str | None = None, doc_class: str = "") -> DocumentMeta:
        source = Path(source)
        return self.add_bytes(source.read_bytes(), name or source.name, doc_class)

    def add_bytes(self, data: bytes, name: str, doc_class: str = "") -> DocumentMeta:
        sha = hashlib.sha256(data).hexdigest()
        doc_id = sha[:16]
        doc_dir = self.docs_dir / doc_id
        (doc_dir / "renders").mkdir(parents=True, exist_ok=True)
        (doc_dir / "results").mkdir(parents=True, exist_ok=True)
        pdf_path = doc_dir / "source.pdf"
        if not pdf_path.exists():
            pdf_path.write_bytes(data)

        meta_path = doc_dir / "meta.json"
        if meta_path.exists():
            meta = DocumentMeta(**json.loads(meta_path.read_text()))
            if doc_class and not meta.doc_class:
                meta.doc_class = doc_class
                meta_path.write_text(json.dumps(meta.as_dict(), indent=2))
            return meta

        meta = DocumentMeta(
            doc_id=doc_id,
            name=name,
            pages=page_count(pdf_path),
            size_bytes=len(data),
            sha256=sha,
            added_at=_now(),
            doc_class=doc_class,
        )
        meta_path.write_text(json.dumps(meta.as_dict(), indent=2))
        return meta

    def list_documents(self) -> list[DocumentMeta]:
        out = []
        for meta_path in sorted(self.docs_dir.glob("*/meta.json")):
            try:
                out.append(DocumentMeta(**json.loads(meta_path.read_text())))
            except Exception:  # pragma: no cover - skip corrupt entries
                continue
        return sorted(out, key=lambda m: m.added_at, reverse=True)

    def get_document(self, doc_id: str) -> DocumentMeta:
        meta_path = self.docs_dir / doc_id / "meta.json"
        if not meta_path.exists():
            raise KeyError(f"unknown document: {doc_id}")
        return DocumentMeta(**json.loads(meta_path.read_text()))

    def update_document(self, meta: DocumentMeta) -> DocumentMeta:
        (self.docs_dir / meta.doc_id / "meta.json").write_text(json.dumps(meta.as_dict(), indent=2))
        return meta

    def delete_document(self, doc_id: str) -> None:
        shutil.rmtree(self.docs_dir / doc_id, ignore_errors=True)

    def pdf_path(self, doc_id: str) -> Path:
        path = self.docs_dir / doc_id / "source.pdf"
        if not path.exists():
            raise KeyError(f"unknown document: {doc_id}")
        return path

    def geometry(self, doc_id: str):
        return page_geometry(self.pdf_path(doc_id))

    # -- renders ---------------------------------------------------------

    def render(self, doc_id: str, page: int, scale: float = 2.0) -> Path:
        out = self.docs_dir / doc_id / "renders" / f"p{page}@{scale:g}.png"
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(render_page(self.pdf_path(doc_id), page, scale))
        return out

    # -- results ---------------------------------------------------------

    def result_path(self, doc_id: str, key: str) -> Path:
        return self.docs_dir / doc_id / "results" / f"{key}.json"

    def save_result(self, result: ParseResult, key: str) -> Path:
        path = self.result_path(result.doc_id, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.model_dump_json())
        return path

    def load_result(self, doc_id: str, key: str) -> ParseResult | None:
        path = self.result_path(doc_id, key)
        if not path.exists():
            return None
        try:
            return ParseResult.model_validate_json(path.read_text())
        except Exception:  # pragma: no cover - stale schema
            return None

    def list_results(self, doc_id: str) -> list[dict[str, Any]]:
        out = []
        results_dir = self.docs_dir / doc_id / "results"
        for path in sorted(results_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:
                continue
            out.append(
                {
                    "key": path.stem,
                    "parser_id": data.get("parser_id"),
                    "parser_name": data.get("parser_name"),
                    "status": data.get("status"),
                    "error": data.get("error"),
                    "duration_s": data.get("duration_s"),
                    "created_at": data.get("created_at"),
                    "options": data.get("options", {}),
                    "n_pages": len(data.get("pages", [])),
                }
            )
        return out

    def delete_result(self, doc_id: str, key: str) -> None:
        self.result_path(doc_id, key).unlink(missing_ok=True)

    # -- ground truth ----------------------------------------------------

    def ground_truth_path(self, doc_id: str) -> Path:
        return self.docs_dir / doc_id / "ground_truth.json"

    def get_ground_truth(self, doc_id: str) -> dict[str, Any] | None:
        path = self.ground_truth_path(doc_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def set_ground_truth(self, doc_id: str, data: dict[str, Any]) -> None:
        self.ground_truth_path(doc_id).write_text(json.dumps(data, indent=2))
