"""PyMuPDF (fitz) adapter.

Fast, native text extraction with word/line/block granularity and a built-in
table finder. PyMuPDF already uses a top-left origin, so boxes pass through
unchanged.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument, Table, TableCell
from .base import Option, PdfParser, select_pages


class PyMuPDFParser(PdfParser):
    id = "pymupdf"
    name = "PyMuPDF"
    kind = "local"
    description = (
        "Native text extraction via MuPDF. Very fast, gives word/line/block boxes and "
        "has a geometric table finder. No layout model — reading order is the PDF's own."
    )
    homepage = "https://pymupdf.readthedocs.io"
    tags = ("text-layer", "fast", "tables", "bboxes")
    requires = ("pymupdf",)
    options = (
        Option("find_tables", "bool", True, help="Run MuPDF's geometric table finder."),
        Option("sort_blocks", "bool", True, help="Sort blocks into top-to-bottom reading order."),
        Option(
            "table_strategy",
            "choice",
            "lines_strict",
            choices=["lines", "lines_strict", "text"],
            help="Table detection strategy for horizontal and vertical lines.",
        ),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        import pymupdf

        opts = self.resolved_options(options)
        doc = pymupdf.open(pdf_path)
        try:
            wanted = select_pages(doc.page_count, pages)
            out_pages: list[PageResult] = []
            timings: dict[int, float] = {}
            warnings: list[str] = []

            for page_no in wanted:
                started = time.perf_counter()
                page = doc[page_no - 1]
                rect = page.rect
                result = PageResult(
                    page_number=page_no,
                    width=rect.width,
                    height=rect.height,
                    rotation=page.rotation,
                    text=page.get_text("text"),
                )

                order = 0
                # Words
                for w in page.get_text("words"):
                    x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                    result.blocks.append(
                        Block(
                            id=f"p{page_no}-w{order}",
                            page=page_no,
                            layer="word",
                            text=word,
                            bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                            order=order,
                        )
                    )
                    order += 1

                # Lines and blocks from the structured dict
                data = page.get_text("dict", sort=opts["sort_blocks"])
                for bi, blk in enumerate(data.get("blocks", [])):
                    if blk.get("type") != 0:  # image block
                        bbox = blk.get("bbox")
                        if bbox:
                            result.blocks.append(
                                Block(
                                    id=f"p{page_no}-img{bi}",
                                    page=page_no,
                                    layer="block",
                                    kind="figure",
                                    bbox=BBox(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3]),
                                    order=bi,
                                )
                            )
                        continue
                    block_text: list[str] = []
                    for li, line in enumerate(blk.get("lines", [])):
                        line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                        block_text.append(line_text)
                        lb = line.get("bbox")
                        sizes = [s.get("size") for s in line.get("spans", []) if s.get("size")]
                        fonts = [s.get("font") for s in line.get("spans", []) if s.get("font")]
                        if lb:
                            result.blocks.append(
                                Block(
                                    id=f"p{page_no}-b{bi}-l{li}",
                                    page=page_no,
                                    layer="line",
                                    text=line_text,
                                    bbox=BBox(x0=lb[0], y0=lb[1], x1=lb[2], y1=lb[3]),
                                    order=bi * 1000 + li,
                                    meta={
                                        "font_size": round(sum(sizes) / len(sizes), 2) if sizes else None,
                                        "font": fonts[0] if fonts else None,
                                    },
                                )
                            )
                    bb = blk.get("bbox")
                    if bb:
                        result.blocks.append(
                            Block(
                                id=f"p{page_no}-b{bi}",
                                page=page_no,
                                layer="block",
                                text="\n".join(block_text),
                                bbox=BBox(x0=bb[0], y0=bb[1], x1=bb[2], y1=bb[3]),
                                order=bi,
                            )
                        )

                if opts["find_tables"]:
                    try:
                        finder = page.find_tables(strategy=opts["table_strategy"])
                        for ti, tab in enumerate(finder.tables):
                            result.tables.append(_convert_table(page_no, ti, tab))
                            bx = tab.bbox
                            result.blocks.append(
                                Block(
                                    id=f"p{page_no}-t{ti}",
                                    page=page_no,
                                    layer="table",
                                    kind="table",
                                    bbox=BBox(x0=bx[0], y0=bx[1], x1=bx[2], y1=bx[3]),
                                    text=f"table {tab.row_count}x{tab.col_count}",
                                    order=ti,
                                )
                            )
                            for ci, cell in enumerate(tab.cells):
                                if cell is None:
                                    continue
                                result.blocks.append(
                                    Block(
                                        id=f"p{page_no}-t{ti}-c{ci}",
                                        page=page_no,
                                        layer="cell",
                                        kind="table",
                                        bbox=BBox(x0=cell[0], y0=cell[1], x1=cell[2], y1=cell[3]),
                                        order=ci,
                                    )
                                )
                    except Exception as exc:  # pragma: no cover - depends on document
                        warnings.append(f"table finder failed on page {page_no}: {exc}")

                timings[page_no] = time.perf_counter() - started
                out_pages.append(result)

            return ParsedDocument(pages=out_pages, per_page_s=timings, warnings=warnings)
        finally:
            doc.close()


def _convert_table(page_no: int, index: int, tab: Any) -> Table:
    rows = tab.extract()
    bx = tab.bbox
    cells: list[TableCell] = []
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cells.append(TableCell(row=r, col=c, text=(value or "").strip(), is_header=(r == 0)))
    return Table(
        id=f"p{page_no}-t{index}",
        page=page_no,
        bbox=BBox(x0=bx[0], y0=bx[1], x1=bx[2], y1=bx[3]),
        n_rows=len(rows),
        n_cols=max((len(r) for r in rows), default=0),
        cells=cells,
    )
