"""pdfplumber adapter.

Built on pdfminer.six, with word/line clustering and a well-regarded table
extractor that can key off ruling lines or text alignment. pdfplumber reports
``top``/``bottom`` from the top of the page, so boxes map directly onto our
convention.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument, Table, TableCell
from .base import Option, PdfParser, select_pages


class PdfplumberParser(PdfParser):
    id = "pdfplumber"
    name = "pdfplumber"
    kind = "local"
    description = (
        "pdfminer.six with word/line clustering and a configurable table extractor. "
        "Slower than PyMuPDF but its 'text' table strategy handles borderless tables, "
        "which is the common case in bank statements."
    )
    homepage = "https://github.com/jsvine/pdfplumber"
    tags = ("text-layer", "tables", "bboxes", "borderless-tables")
    requires = ("pdfplumber",)
    options = (
        Option("extract_tables", "bool", True),
        Option(
            "table_strategy",
            "choice",
            "lines",
            choices=["lines", "text", "explicit"],
            help="Both vertical and horizontal strategy. 'text' finds borderless tables.",
        ),
        Option("x_tolerance", "float", 3.0, help="Horizontal gap (pt) that separates two words."),
        Option("y_tolerance", "float", 3.0, help="Vertical tolerance for grouping chars into a line."),
        Option("keep_blank_chars", "bool", False),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        import pdfplumber

        opts = self.resolved_options(options)
        warnings: list[str] = []
        timings: dict[int, float] = {}
        out_pages: list[PageResult] = []

        table_settings = {
            "vertical_strategy": opts["table_strategy"],
            "horizontal_strategy": opts["table_strategy"],
        }
        word_kwargs = {
            "x_tolerance": opts["x_tolerance"],
            "y_tolerance": opts["y_tolerance"],
            "keep_blank_chars": opts["keep_blank_chars"],
        }

        with pdfplumber.open(pdf_path) as pdf:
            wanted = select_pages(len(pdf.pages), pages)
            for page_no in wanted:
                started = time.perf_counter()
                page = pdf.pages[page_no - 1]
                result = PageResult(
                    page_number=page_no,
                    width=float(page.width),
                    height=float(page.height),
                    rotation=int(page.rotation or 0),
                    text=page.extract_text(**word_kwargs) or "",
                )

                for i, w in enumerate(page.extract_words(**word_kwargs)):
                    result.blocks.append(
                        Block(
                            id=f"p{page_no}-w{i}",
                            page=page_no,
                            layer="word",
                            text=w["text"],
                            bbox=BBox(x0=w["x0"], y0=w["top"], x1=w["x1"], y1=w["bottom"]),
                            order=i,
                            meta={"upright": w.get("upright")},
                        )
                    )

                try:
                    lines = page.extract_text_lines(**word_kwargs)
                except Exception as exc:  # pragma: no cover - older pdfplumber
                    lines = []
                    warnings.append(f"extract_text_lines failed on page {page_no}: {exc}")
                for i, line in enumerate(lines):
                    result.blocks.append(
                        Block(
                            id=f"p{page_no}-l{i}",
                            page=page_no,
                            layer="line",
                            text=line["text"],
                            bbox=BBox(x0=line["x0"], y0=line["top"], x1=line["x1"], y1=line["bottom"]),
                            order=i,
                        )
                    )

                if opts["extract_tables"]:
                    try:
                        found = page.find_tables(table_settings=table_settings)
                    except Exception as exc:
                        found = []
                        warnings.append(f"find_tables failed on page {page_no}: {exc}")
                    for ti, tab in enumerate(found):
                        data = tab.extract()
                        cells: list[TableCell] = []
                        for r, row in enumerate(data):
                            for c, value in enumerate(row):
                                cells.append(
                                    TableCell(row=r, col=c, text=(value or "").strip(), is_header=(r == 0))
                                )
                        bx = tab.bbox
                        result.tables.append(
                            Table(
                                id=f"p{page_no}-t{ti}",
                                page=page_no,
                                bbox=BBox(x0=bx[0], y0=bx[1], x1=bx[2], y1=bx[3]),
                                n_rows=len(data),
                                n_cols=max((len(r) for r in data), default=0),
                                cells=cells,
                                meta={"strategy": opts["table_strategy"]},
                            )
                        )
                        result.blocks.append(
                            Block(
                                id=f"p{page_no}-t{ti}",
                                page=page_no,
                                layer="table",
                                kind="table",
                                text=f"table {len(data)}x{max((len(r) for r in data), default=0)}",
                                bbox=BBox(x0=bx[0], y0=bx[1], x1=bx[2], y1=bx[3]),
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

                timings[page_no] = time.perf_counter() - started
                out_pages.append(result)

        return ParsedDocument(pages=out_pages, per_page_s=timings, warnings=warnings)
