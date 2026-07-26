"""pdfminer.six adapter.

The reference pure-Python implementation, and the layout engine underneath
pdfplumber. Exposed separately because its `LAParams` knobs (line/char/word
margins) change block grouping dramatically — useful for seeing how sensitive
"paragraph" detection is on a given document class.

pdfminer uses a bottom-left origin, so every box is flipped on the way in.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..models import BBox, Block, PageResult, ParsedDocument
from .base import Option, PdfParser, select_pages


class PdfminerParser(PdfParser):
    id = "pdfminer"
    name = "pdfminer.six"
    kind = "local"
    description = (
        "Pure-Python layout analysis with tunable LAParams. No table support. Its block "
        "grouping is the thing to watch: char/line/word margins decide whether statement "
        "columns merge into one paragraph or stay apart."
    )
    homepage = "https://pdfminersix.readthedocs.io"
    tags = ("text-layer", "layout-params", "bboxes")
    requires = ("pdfminer",)
    options = (
        Option("char_margin", "float", 2.0, help="Chars closer than this (x avg width) join a line."),
        Option("line_margin", "float", 0.5, help="Lines closer than this (x height) join a text box."),
        Option("word_margin", "float", 0.1, help="Gap that inserts a space between chars."),
        Option("detect_vertical", "bool", False),
        Option("all_texts", "bool", False, help="Also run layout analysis inside figures."),
    )

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LAParams, LTChar, LTTextBox, LTTextLine

        opts = self.resolved_options(options)
        laparams = LAParams(
            char_margin=opts["char_margin"],
            line_margin=opts["line_margin"],
            word_margin=opts["word_margin"],
            detect_vertical=opts["detect_vertical"],
            all_texts=opts["all_texts"],
        )
        wanted = sorted(pages) if pages else None
        page_numbers = None if wanted is None else [p - 1 for p in wanted]

        out_pages: list[PageResult] = []
        timings: dict[int, float] = {}
        started = time.perf_counter()

        for i, layout in enumerate(
            extract_pages(str(pdf_path), laparams=laparams, page_numbers=page_numbers)
        ):
            # LTPage.pageid counts the pages pdfminer actually processed, so with
            # a page filter it restarts at 1. Map back to the real page number.
            page_no = wanted[i] if wanted is not None else i + 1
            height = layout.height
            result = PageResult(
                page_number=page_no,
                width=layout.width,
                height=height,
                rotation=int(getattr(layout, "rotate", 0) or 0),
            )
            texts: list[str] = []
            bi = li = wi = 0
            for element in layout:
                if not isinstance(element, LTTextBox):
                    continue
                box_text = element.get_text()
                texts.append(box_text)
                result.blocks.append(
                    Block(
                        id=f"p{page_no}-b{bi}",
                        page=page_no,
                        layer="block",
                        text=box_text.rstrip("\n"),
                        bbox=BBox.from_bottom_left(*element.bbox, page_height=height),
                        order=bi,
                    )
                )
                for line in element:
                    if not isinstance(line, LTTextLine):
                        continue
                    result.blocks.append(
                        Block(
                            id=f"p{page_no}-l{li}",
                            page=page_no,
                            layer="line",
                            text=line.get_text().rstrip("\n"),
                            bbox=BBox.from_bottom_left(*line.bbox, page_height=height),
                            order=li,
                        )
                    )
                    li += 1
                    for word, bbox in _words_from_line(line, LTChar):
                        result.blocks.append(
                            Block(
                                id=f"p{page_no}-w{wi}",
                                page=page_no,
                                layer="word",
                                text=word,
                                bbox=BBox.from_bottom_left(*bbox, page_height=height),
                                order=wi,
                            )
                        )
                        wi += 1
                bi += 1
            result.text = "".join(texts)
            out_pages.append(result)
            now = time.perf_counter()
            timings[page_no] = now - started
            started = now

        return ParsedDocument(pages=out_pages, per_page_s=timings)


def _words_from_line(line: Any, ltchar_cls: type) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Split an LTTextLine into whitespace-delimited words with a merged bbox."""
    words: list[tuple[str, tuple[float, float, float, float]]] = []
    chars: list[Any] = []

    def flush() -> None:
        if not chars:
            return
        text = "".join(c.get_text() for c in chars)
        if text.strip():
            words.append(
                (
                    text,
                    (
                        min(c.x0 for c in chars),
                        min(c.y0 for c in chars),
                        max(c.x1 for c in chars),
                        max(c.y1 for c in chars),
                    ),
                )
            )
        chars.clear()

    for obj in line:
        if isinstance(obj, ltchar_cls) and obj.get_text().strip():
            chars.append(obj)
        else:
            flush()
    flush()
    return words
