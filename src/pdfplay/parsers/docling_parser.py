"""Docling adapter.

Docling runs a real layout model (DocLayNet-style region classification) plus
TableFormer for table structure, and can fall back to OCR for scanned pages.
It runs locally but pulls in torch and downloads model weights on first use,
so it lives behind the ``docling`` extra.

Docling reports provenance boxes in its own coordinate origin; we normalize
through ``BoundingBox.to_top_left_origin``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument, Table, TableCell
from .base import Option, PdfParser


# Docling labels -> our `kind` vocabulary.
_LABEL_KIND = {
    "title": "title",
    "section_header": "title",
    "page_header": "header",
    "page_footer": "footer",
    "list_item": "list",
    "table": "table",
    "picture": "figure",
    "caption": "caption",
    "formula": "formula",
    "code": "text",
    "text": "text",
    "paragraph": "text",
    "key_value_region": "key_value",
}


class DoclingParser(PdfParser):
    id = "docling"
    name = "Docling"
    kind = "local"
    description = (
        "IBM's Docling: layout model + TableFormer table structure + optional OCR. "
        "Runs locally but downloads model weights on first use. Gives labelled regions "
        "(title/header/footer/table/list), which is what you want for statement sectioning."
    )
    homepage = "https://github.com/docling-project/docling"
    tags = ("layout-model", "tables", "ocr", "bboxes", "markdown")
    requires = ("docling",)
    extra = "docling"
    cost_hint = "local compute (CPU/GPU), model download on first run"
    options = (
        Option("do_ocr", "bool", False, help="OCR pages with no usable text layer."),
        Option("do_table_structure", "bool", True, help="Run TableFormer for cell structure."),
        Option(
            "table_mode",
            "choice",
            "accurate",
            choices=["fast", "accurate"],
            help="TableFormer speed/accuracy tradeoff.",
        ),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption

        opts = self.resolved_options(options)
        pipeline = PdfPipelineOptions()
        pipeline.do_ocr = bool(opts["do_ocr"])
        pipeline.do_table_structure = bool(opts["do_table_structure"])
        if pipeline.do_table_structure:
            pipeline.table_structure_options.mode = (
                TableFormerMode.ACCURATE if opts["table_mode"] == "accurate" else TableFormerMode.FAST
            )

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
        )

        started = time.perf_counter()
        convert_kwargs: dict[str, Any] = {}
        if pages:
            convert_kwargs["page_range"] = (min(pages), max(pages))
        try:
            conv = converter.convert(str(pdf_path), **convert_kwargs)
        except TypeError:
            # Older docling without page_range support.
            conv = converter.convert(str(pdf_path))
        doc = conv.document
        elapsed = time.perf_counter() - started

        wanted = set(pages) if pages else None
        page_sizes: dict[int, tuple[float, float]] = {}
        for page_no, page_item in (getattr(doc, "pages", None) or {}).items():
            size = getattr(page_item, "size", None)
            if size is not None:
                page_sizes[int(page_no)] = (float(size.width), float(size.height))

        results: dict[int, PageResult] = {}

        def page_for(page_no: int) -> PageResult:
            if page_no not in results:
                width, height = page_sizes.get(page_no, (612.0, 792.0))
                results[page_no] = PageResult(page_number=page_no, width=width, height=height)
            return results[page_no]

        counter = 0
        for item, _level in doc.iterate_items():
            label = str(getattr(item, "label", "") or "")
            label = label.split(".")[-1].lower()
            text = getattr(item, "text", "") or ""
            for prov in getattr(item, "prov", []) or []:
                page_no = int(getattr(prov, "page_no", 1))
                if wanted is not None and page_no not in wanted:
                    continue
                page = page_for(page_no)
                bbox = _to_bbox(getattr(prov, "bbox", None), page.height)
                page.blocks.append(
                    Block(
                        id=f"p{page_no}-d{counter}",
                        page=page_no,
                        layer="region",
                        kind=_LABEL_KIND.get(label, "text"),
                        text=text,
                        bbox=bbox,
                        order=counter,
                        meta={"docling_label": label},
                    )
                )
                counter += 1
            if label == "table":
                _add_table(item, page_for, wanted)

        for page in results.values():
            page.text = "\n".join(b.text for b in page.blocks if b.text)

        markdown = None
        try:
            markdown = doc.export_to_markdown()
        except Exception:  # pragma: no cover - export is best effort
            pass

        ordered = [results[k] for k in sorted(results)]
        per_page = {p.page_number: elapsed / max(1, len(ordered)) for p in ordered}
        return ParsedDocument(pages=ordered, markdown=markdown, per_page_s=per_page)


def _to_bbox(bbox: Any, page_height: float) -> BBox | None:
    if bbox is None:
        return None
    try:
        tl = bbox.to_top_left_origin(page_height=page_height)
        return BBox(x0=float(tl.l), y0=float(tl.t), x1=float(tl.r), y1=float(tl.b)).normalized()
    except Exception:
        try:
            return BBox(x0=float(bbox.l), y0=float(bbox.t), x1=float(bbox.r), y1=float(bbox.b)).normalized()
        except Exception:  # pragma: no cover - unknown bbox shape
            return None


def _add_table(item: Any, page_for: Any, wanted: set[int] | None) -> None:
    prov = (getattr(item, "prov", None) or [None])[0]
    if prov is None:
        return
    page_no = int(getattr(prov, "page_no", 1))
    if wanted is not None and page_no not in wanted:
        return
    page = page_for(page_no)
    data = getattr(item, "data", None)
    cells_in = list(getattr(data, "table_cells", []) or []) if data is not None else []
    cells: list[TableCell] = []
    n_rows = int(getattr(data, "num_rows", 0) or 0)
    n_cols = int(getattr(data, "num_cols", 0) or 0)
    for cell in cells_in:
        row = int(getattr(cell, "start_row_offset_idx", 0) or 0)
        col = int(getattr(cell, "start_col_offset_idx", 0) or 0)
        row_end = int(getattr(cell, "end_row_offset_idx", row + 1) or row + 1)
        col_end = int(getattr(cell, "end_col_offset_idx", col + 1) or col + 1)
        cells.append(
            TableCell(
                row=row,
                col=col,
                row_span=max(1, row_end - row),
                col_span=max(1, col_end - col),
                text=(getattr(cell, "text", "") or "").strip(),
                bbox=_to_bbox(getattr(cell, "bbox", None), page.height),
                is_header=bool(getattr(cell, "column_header", False)),
            )
        )
        n_rows = max(n_rows, row_end)
        n_cols = max(n_cols, col_end)
    page.tables.append(
        Table(
            id=f"p{page_no}-t{len(page.tables)}",
            page=page_no,
            bbox=_to_bbox(getattr(prov, "bbox", None), page.height),
            n_rows=n_rows,
            n_cols=n_cols,
            cells=cells,
        )
    )
