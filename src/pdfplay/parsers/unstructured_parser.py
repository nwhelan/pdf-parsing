"""Unstructured.io adapter (local `partition_pdf`).

Unstructured classifies elements (Title / NarrativeText / Table / ListItem /
Header / Footer) and can run a detectron/YOLOX layout model in `hi_res` mode.
Its coordinates come back in a pixel space sized by the rendering DPI, so we
rescale into PDF points using the true page size.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument
from ..geometry import page_geometry
from .base import Option, PdfParser


_CATEGORY_KIND = {
    "Title": "title",
    "Header": "header",
    "Footer": "footer",
    "PageNumber": "footer",
    "ListItem": "list",
    "Table": "table",
    "Image": "figure",
    "Figure": "figure",
    "FigureCaption": "caption",
    "Formula": "formula",
    "Address": "text",
    "EmailAddress": "text",
    "NarrativeText": "text",
    "UncategorizedText": "text",
}


class UnstructuredParser(PdfParser):
    id = "unstructured"
    name = "Unstructured"
    kind = "local"
    description = (
        "Element classification (Title/NarrativeText/Table/ListItem/Header/Footer) with an "
        "optional layout model in hi_res mode. Good at sectioning; table HTML quality varies."
    )
    homepage = "https://github.com/Unstructured-IO/unstructured"
    tags = ("layout-model", "classification", "ocr", "bboxes")
    requires = ("unstructured",)
    extra = "unstructured"
    cost_hint = "local compute; hi_res needs a layout model + poppler/tesseract"
    options = (
        Option(
            "strategy",
            "choice",
            "hi_res",
            choices=["auto", "fast", "hi_res", "ocr_only"],
            help="'fast' uses only the text layer; 'hi_res' runs the layout model.",
        ),
        Option("infer_table_structure", "bool", True),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        from unstructured.partition.pdf import partition_pdf

        opts = self.resolved_options(options)
        geometry = page_geometry(pdf_path)
        wanted = set(pages) if pages else None

        started = time.perf_counter()
        elements = partition_pdf(
            filename=str(pdf_path),
            strategy=opts["strategy"],
            infer_table_structure=bool(opts["infer_table_structure"]),
        )
        elapsed = time.perf_counter() - started

        results: dict[int, PageResult] = {}
        for i, element in enumerate(elements):
            meta = getattr(element, "metadata", None)
            page_no = int(getattr(meta, "page_number", 1) or 1)
            if wanted is not None and page_no not in wanted:
                continue
            geo = geometry.get(page_no)
            if page_no not in results:
                results[page_no] = PageResult(
                    page_number=page_no,
                    width=geo.width if geo else 612.0,
                    height=geo.height if geo else 792.0,
                    rotation=geo.rotation if geo else 0,
                )
            page = results[page_no]
            category = getattr(element, "category", None) or type(element).__name__
            page.blocks.append(
                Block(
                    id=f"p{page_no}-u{i}",
                    page=page_no,
                    layer="region",
                    kind=_CATEGORY_KIND.get(category, "text"),
                    text=(getattr(element, "text", "") or ""),
                    bbox=_coords_to_bbox(getattr(meta, "coordinates", None), page.width, page.height),
                    order=i,
                    meta={
                        "category": category,
                        "table_html": getattr(meta, "text_as_html", None),
                    },
                )
            )

        for page in results.values():
            page.text = "\n".join(b.text for b in page.blocks if b.text)

        ordered = [results[k] for k in sorted(results)]
        per_page = {p.page_number: elapsed / max(1, len(ordered)) for p in ordered}
        return ParsedDocument(pages=ordered, per_page_s=per_page)


def _coords_to_bbox(coords: Any, page_width: float, page_height: float) -> BBox | None:
    if coords is None:
        return None
    points = getattr(coords, "points", None)
    if not points:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    system = getattr(coords, "system", None)
    layout_w = float(getattr(system, "width", 0) or 0)
    layout_h = float(getattr(system, "height", 0) or 0)
    sx = page_width / layout_w if layout_w else 1.0
    sy = page_height / layout_h if layout_h else 1.0
    box = BBox(x0=min(xs) * sx, y0=min(ys) * sy, x1=max(xs) * sx, y1=max(ys) * sy)
    # Some coordinate systems are bottom-left ("PixelSpace" is top-left).
    if str(type(system).__name__).lower().startswith("pointspace") and layout_h:
        box = BBox.from_bottom_left(box.x0, page_height - box.y1, box.x1, page_height - box.y0, page_height)
    return box.normalized()
