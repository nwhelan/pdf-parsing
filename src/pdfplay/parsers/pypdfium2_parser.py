"""pypdfium2 adapter — PDFium's text layer.

PDFium is the extraction engine inside Chrome. It has no notion of blocks or
tables; it exposes text "rects" that are roughly line fragments. Useful as a
third opinion on the raw text layer, and as the fastest correctness baseline.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument
from .base import PdfParser, select_pages


class Pypdfium2Parser(PdfParser):
    id = "pypdfium2"
    name = "pypdfium2 (PDFium)"
    kind = "local"
    description = (
        "Chrome's PDF engine. Text rects only — no blocks, no tables. Fast and a good "
        "reference for 'what does the text layer actually contain'."
    )
    homepage = "https://github.com/pypdfium2-team/pypdfium2"
    tags = ("text-layer", "fast", "bboxes")
    requires = ("pypdfium2",)

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        import pypdfium2 as pdfium

        pdf = pdfium.PdfDocument(str(pdf_path))
        try:
            wanted = select_pages(len(pdf), pages)
            out_pages: list[PageResult] = []
            timings: dict[int, float] = {}

            for page_no in wanted:
                started = time.perf_counter()
                page = pdf[page_no - 1]
                height = page.get_height()
                textpage = page.get_textpage()
                result = PageResult(
                    page_number=page_no,
                    width=page.get_width(),
                    height=height,
                    rotation=page.get_rotation(),
                    text=textpage.get_text_bounded(),
                )
                for i in range(textpage.count_rects()):
                    left, bottom, right, top = textpage.get_rect(i)
                    text = textpage.get_text_bounded(left=left, bottom=bottom, right=right, top=top)
                    result.blocks.append(
                        Block(
                            id=f"p{page_no}-r{i}",
                            page=page_no,
                            layer="line",
                            text=text,
                            bbox=BBox.from_bottom_left(left, bottom, right, top, height),
                            order=i,
                        )
                    )
                textpage.close()
                timings[page_no] = time.perf_counter() - started
                out_pages.append(result)

            return ParsedDocument(pages=out_pages, per_page_s=timings)
        finally:
            pdf.close()
