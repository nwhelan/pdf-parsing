"""Page geometry + rasterization, both backed by pypdfium2."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class PageGeometry:
    number: int  # 1-based
    width: float  # points
    height: float  # points
    rotation: int


@lru_cache(maxsize=64)
def _geometry_cached(path: str, mtime: float) -> tuple[PageGeometry, ...]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        return tuple(
            PageGeometry(
                number=i + 1,
                width=pdf[i].get_width(),
                height=pdf[i].get_height(),
                rotation=pdf[i].get_rotation(),
            )
            for i in range(len(pdf))
        )
    finally:
        pdf.close()


def page_geometry(pdf_path: Path | str) -> dict[int, PageGeometry]:
    """Map of 1-based page number -> geometry in PDF points."""
    p = Path(pdf_path)
    geo = _geometry_cached(str(p), p.stat().st_mtime)
    return {g.number: g for g in geo}


def page_count(pdf_path: Path | str) -> int:
    return len(page_geometry(pdf_path))


def render_page(pdf_path: Path | str, page_number: int, scale: float = 2.0) -> bytes:
    """Render one 1-based page to PNG bytes at ``scale`` px per point."""
    import io

    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        page = pdf[page_number - 1]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        pdf.close()


def render_page_image(pdf_path: Path | str, page_number: int, scale: float = 2.0):
    """Render one 1-based page to a PIL image (used by the OCR + vision adapters)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return pdf[page_number - 1].render(scale=scale).to_pil()
    finally:
        pdf.close()
