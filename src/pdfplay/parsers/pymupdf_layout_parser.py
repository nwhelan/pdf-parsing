"""PyMuPDF Layout adapter, via ``pymupdf4llm.to_markdown``.

``pymupdf4llm`` has two engines behind the same call:

``layout``
    The ONNX layout model shipped in the ``pymupdf-layout`` package. It
    classifies regions (page-header / section-header / text / table /
    page-footer), emits real Markdown tables, and will OCR a page that has no
    usable text layer. This is the interesting one for statements: it is the
    only local parser here that reconstructs the table *as a table*.

``classic``
    The original heuristic Markdown writer. No layout model, no OCR, but it
    returns word-level boxes — useful as a controlled comparison, since the
    library, the text layer, and the Markdown target are all held constant and
    only the layout engine changes.

Both engines report boxes in PDF points with a top-left origin, so nothing
needs converting.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
from pathlib import Path
from typing import Any

from ..geometry import page_geometry
from ..markdown_tables import tables_from_markdown
from ..models import BBox, Block, PageResult, ParsedDocument
from .base import Option, PdfParser, select_pages

# `pymupdf4llm.use_layout()` flips module-global state and swaps which function
# `to_markdown` dispatches to, so two calls must not overlap.
_ENGINE_LOCK = threading.Lock()

_CLASS_KIND = {
    "page-header": "header",
    "page-footer": "footer",
    "section-header": "title",
    "title": "title",
    "text": "text",
    "table": "table",
    "figure": "figure",
    "picture": "figure",
    "image": "figure",
    "caption": "caption",
    "list": "list",
    "list-item": "list",
    "formula": "formula",
    "code": "text",
    "footnote": "text",
}


class PyMuPDFLayoutParser(PdfParser):
    id = "pymupdf-layout"
    name = "PyMuPDF Layout (4LLM)"
    kind = "local"
    description = (
        "pymupdf4llm.to_markdown with the pymupdf-layout ONNX model: classified regions, "
        "real Markdown tables, and OCR fallback for pages with no text layer. Switch the "
        "engine option to 'classic' to run the same call without the layout model."
    )
    homepage = "https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/"
    tags = ("layout-model", "markdown", "tables", "ocr", "bboxes")
    requires = ("pymupdf4llm",)
    extra = "layout"
    cost_hint = "local compute; the layout engine runs an ONNX model per page"
    options = (
        Option(
            "engine",
            "choice",
            "layout",
            choices=["layout", "classic"],
            help="'layout' uses the pymupdf-layout model; 'classic' is the original heuristic writer.",
        ),
        Option("use_ocr", "bool", True, help="Layout only: OCR pages with no usable text layer."),
        Option("force_ocr", "bool", False, help="Layout only: OCR every page, even if text exists."),
        Option("ocr_language", "str", "eng", help="Layout only: Tesseract language code."),
        Option("ocr_dpi", "int", 300, help="Layout only: rasterization DPI for OCR."),
        Option("header", "bool", True, help="Layout only: keep page headers in the Markdown."),
        Option("footer", "bool", True, help="Layout only: keep page footers in the Markdown."),
        Option("ignore_code", "bool", False, help="Don't emit mono-spaced text as code spans."),
        Option(
            "table_strategy",
            "choice",
            "lines_strict",
            choices=["lines", "lines_strict", "text"],
            help="Classic only: how tables are detected.",
        ),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        import pymupdf4llm

        opts = self.resolved_options(options)
        layout = opts["engine"] == "layout"
        geometry = page_geometry(pdf_path)
        wanted = select_pages(len(geometry), pages)
        warnings: list[str] = []

        if layout and not _layout_engine_available():
            raise RuntimeError(
                "the layout engine needs the pymupdf-layout package "
                "(pip install -e '.[layout]'); or set engine=classic"
            )

        kwargs: dict[str, Any] = {
            "page_chunks": True,
            "pages": [p - 1 for p in wanted],  # pymupdf4llm counts pages from 0
            "ignore_code": bool(opts["ignore_code"]),
            "show_progress": False,
        }
        if layout:
            kwargs.update(
                use_ocr=bool(opts["use_ocr"]),
                force_ocr=bool(opts["force_ocr"]),
                ocr_language=str(opts["ocr_language"]),
                ocr_dpi=int(opts["ocr_dpi"]),
                header=bool(opts["header"]),
                footer=bool(opts["footer"]),
            )
        else:
            kwargs.update(extract_words=True, table_strategy=opts["table_strategy"])

        started = time.perf_counter()
        chatter = io.StringIO()
        with _ENGINE_LOCK:
            try:
                pymupdf4llm.use_layout(layout)
                # The layout engine narrates to stdout; keep it out of the console
                # and surface it as warnings instead.
                with contextlib.redirect_stdout(chatter):
                    chunks = pymupdf4llm.to_markdown(str(pdf_path), **kwargs)
            finally:
                pymupdf4llm.use_layout(False)
        elapsed = time.perf_counter() - started

        for line in chatter.getvalue().splitlines():
            line = line.strip()
            if line and not line.startswith("==="):
                warnings.append(line)

        out_pages: list[PageResult] = []
        markdown_parts: list[str] = []

        for index, chunk in enumerate(chunks):
            # Chunk metadata carries no page number, so map back by position.
            page_no = wanted[index] if index < len(wanted) else index + 1
            geo = geometry[page_no]
            text = chunk.get("text") or ""
            markdown_parts.append(text)

            page = PageResult(
                page_number=page_no,
                width=geo.width,
                height=geo.height,
                rotation=geo.rotation,
                text=text,
                meta={"engine": opts["engine"], "markdown": text},
            )

            if layout:
                _add_layout_regions(page, chunk, text)
            else:
                _add_classic_words(page, chunk)

            _add_markdown_tables(page, text, layout)
            out_pages.append(page)

        per_page = {p.page_number: elapsed / max(1, len(out_pages)) for p in out_pages}
        return ParsedDocument(
            pages=out_pages,
            markdown="\n\n---\n\n".join(markdown_parts) if markdown_parts else None,
            warnings=warnings,
            per_page_s=per_page,
        )


def _layout_engine_available() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pymupdf.layout") is not None
    except (ImportError, ValueError):
        return False


def _add_layout_regions(page: PageResult, chunk: dict[str, Any], text: str) -> None:
    """Turn the layout model's classified regions into blocks.

    Each region carries ``pos``: the start/end character offsets of that region
    inside the page's Markdown, which is how a region gets its own text.
    """
    for i, box in enumerate(chunk.get("page_boxes") or []):
        bbox = box.get("bbox")
        pos = box.get("pos") or ()
        region_text = ""
        if len(pos) == 2:
            try:
                region_text = text[int(pos[0]) : int(pos[1])].strip()
            except (TypeError, ValueError):
                region_text = ""
        label = str(box.get("class") or "text").lower()
        page.blocks.append(
            Block(
                id=f"p{page.page_number}-r{i}",
                page=page.page_number,
                layer="region",
                kind=_CLASS_KIND.get(label, "text"),
                text=region_text,
                bbox=_bbox(bbox),
                order=int(box.get("index", i)),
                meta={"layout_class": label},
            )
        )


def _add_classic_words(page: PageResult, chunk: dict[str, Any]) -> None:
    """Word boxes and table/figure regions from the classic engine."""
    for i, word in enumerate(chunk.get("words") or []):
        if len(word) < 5:
            continue
        x0, y0, x1, y1, value = word[0], word[1], word[2], word[3], word[4]
        page.blocks.append(
            Block(
                id=f"p{page.page_number}-w{i}",
                page=page.page_number,
                layer="word",
                text=str(value),
                bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                order=i,
            )
        )
    for i, table in enumerate(chunk.get("tables") or []):
        page.blocks.append(
            Block(
                id=f"p{page.page_number}-t{i}",
                page=page.page_number,
                layer="table",
                kind="table",
                text=f"table {table.get('rows', 0)}x{table.get('columns', 0)}",
                bbox=_bbox(table.get("bbox")),
                order=i,
            )
        )
    for i, image in enumerate(chunk.get("images") or []):
        page.blocks.append(
            Block(
                id=f"p{page.page_number}-img{i}",
                page=page.page_number,
                layer="block",
                kind="figure",
                bbox=_bbox(image.get("bbox")),
                order=i,
            )
        )


def _add_markdown_tables(page: PageResult, text: str, layout: bool) -> None:
    """Recover Table objects from the GFM tables in the page Markdown.

    Neither engine hands back cell-level structure — the classic one reports a
    table's bbox and shape, the layout one only classifies the region — but both
    write the cells into the Markdown.
    """
    boxes = [b.bbox for b in page.blocks if b.kind == "table" and b.bbox]
    page.tables.extend(
        tables_from_markdown(
            page.page_number,
            text,
            boxes=boxes,
            meta={"engine": "layout" if layout else "classic"},
        )
    )


def _bbox(raw: Any) -> BBox | None:
    if not raw or len(raw) != 4:
        return None
    try:
        return BBox(x0=float(raw[0]), y0=float(raw[1]), x1=float(raw[2]), y1=float(raw[3])).normalized()
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None
