"""Tesseract OCR adapter.

Rasterizes each page and runs Tesseract over the image, ignoring the text
layer entirely. That makes it the control case: whatever it produces is what a
scanned version of this document would give you, and its per-word confidences
are a cheap signal for how degraded a scan is.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..geometry import page_geometry, render_page_image
from ..models import BBox, Block, PageResult, ParsedDocument
from .base import Option, PdfParser, select_pages


class TesseractParser(PdfParser):
    id = "tesseract"
    name = "Tesseract OCR"
    kind = "local"
    description = (
        "Rasterize + OCR, ignoring the text layer. The control case for scanned statements, "
        "and the only local parser that reports per-word confidence."
    )
    homepage = "https://github.com/tesseract-ocr/tesseract"
    tags = ("ocr", "bboxes", "confidence")
    requires = ("pytesseract",)
    extra = "ocr"
    cost_hint = "local compute; needs the tesseract binary on PATH"
    options = (
        Option("dpi", "int", 200, help="Rasterization DPI. 200-300 is the usual sweet spot."),
        Option("lang", "str", "eng"),
        Option(
            "psm",
            "choice",
            6,
            choices=[3, 4, 6, 11, 12],
            help="Page segmentation mode: 6=uniform block, 4=columns, 11/12=sparse.",
        ),
        Option("min_confidence", "int", 0, help="Drop words below this confidence (0-100)."),
    )

    @classmethod
    def check_availability(cls):
        base = super().check_availability()
        if not base.available:
            return base
        from .base import Availability

        try:
            import pytesseract

            pytesseract.get_tesseract_version()
        except Exception as exc:
            return Availability(False, f"tesseract binary not usable: {exc}")
        return base

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        import pytesseract

        opts = self.resolved_options(options)
        scale = float(opts["dpi"]) / 72.0
        geometry = page_geometry(pdf_path)
        wanted = select_pages(len(geometry), pages)
        config = f"--psm {opts['psm']}"

        out_pages: list[PageResult] = []
        timings: dict[int, float] = {}

        for page_no in wanted:
            started = time.perf_counter()
            geo = geometry[page_no]
            image = render_page_image(pdf_path, page_no, scale=scale)
            data = pytesseract.image_to_data(
                image, lang=opts["lang"], config=config, output_type=pytesseract.Output.DICT
            )
            result = PageResult(
                page_number=page_no, width=geo.width, height=geo.height, rotation=geo.rotation
            )
            px_to_pt = 1.0 / scale
            lines: dict[tuple[int, int, int], list[tuple[str, BBox, float]]] = {}

            for i in range(len(data["text"])):
                text = (data["text"][i] or "").strip()
                if not text:
                    continue
                conf = float(data["conf"][i])
                if conf < float(opts["min_confidence"]):
                    continue
                box = BBox(
                    x0=data["left"][i] * px_to_pt,
                    y0=data["top"][i] * px_to_pt,
                    x1=(data["left"][i] + data["width"][i]) * px_to_pt,
                    y1=(data["top"][i] + data["height"][i]) * px_to_pt,
                )
                result.blocks.append(
                    Block(
                        id=f"p{page_no}-w{i}",
                        page=page_no,
                        layer="word",
                        text=text,
                        bbox=box,
                        order=i,
                        confidence=conf / 100.0,
                    )
                )
                key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
                lines.setdefault(key, []).append((text, box, conf))

            for li, (key, words) in enumerate(sorted(lines.items())):
                boxes = [b for _, b, _ in words]
                confs = [c for _, _, c in words]
                result.blocks.append(
                    Block(
                        id=f"p{page_no}-l{li}",
                        page=page_no,
                        layer="line",
                        text=" ".join(t for t, _, _ in words),
                        bbox=BBox(
                            x0=min(b.x0 for b in boxes),
                            y0=min(b.y0 for b in boxes),
                            x1=max(b.x1 for b in boxes),
                            y1=max(b.y1 for b in boxes),
                        ),
                        order=li,
                        confidence=sum(confs) / len(confs) / 100.0,
                    )
                )

            result.text = "\n".join(b.text for b in result.blocks if b.layer == "line")
            timings[page_no] = time.perf_counter() - started
            out_pages.append(result)

        return ParsedDocument(pages=out_pages, per_page_s=timings)
