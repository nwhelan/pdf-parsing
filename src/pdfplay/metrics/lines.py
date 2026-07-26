"""Line reconstruction.

Parsers disagree about granularity: some give you words, some lines, some
whole labelled regions. Everything downstream (bank-statement scoring, column
analysis, text diffs) wants *lines*, so this module normalizes whatever a
parser produced into a single line representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import BBox, PageResult


@dataclass
class TextLine:
    page: int
    text: str
    bbox: BBox | None = None
    tokens: list[tuple[str, BBox]] = field(default_factory=list)
    source: str = ""  # which layer it came from


def _merge(boxes: list[BBox]) -> BBox | None:
    if not boxes:
        return None
    return BBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _lines_from_words(page: PageResult, y_tolerance: float = 2.0) -> list[TextLine]:
    words = [b for b in page.blocks if b.layer == "word" and b.bbox]
    if not words:
        return []
    words.sort(key=lambda b: (round(b.bbox.cy, 1), b.bbox.x0))
    lines: list[list] = []
    current: list = []
    current_cy: float | None = None
    for w in words:
        cy = w.bbox.cy
        height = max(w.bbox.height, 1.0)
        if current_cy is None or abs(cy - current_cy) <= max(y_tolerance, height * 0.5):
            current.append(w)
            current_cy = cy if current_cy is None else (current_cy + cy) / 2
        else:
            lines.append(current)
            current = [w]
            current_cy = cy
    if current:
        lines.append(current)

    out: list[TextLine] = []
    for group in lines:
        group.sort(key=lambda b: b.bbox.x0)
        out.append(
            TextLine(
                page=page.page_number,
                text=" ".join(b.text for b in group).strip(),
                bbox=_merge([b.bbox for b in group]),
                tokens=[(b.text, b.bbox) for b in group],
                source="word",
            )
        )
    return out


def merge_rows(lines: list[TextLine], overlap: float = 0.6) -> list[TextLine]:
    """Join line fragments that sit on the same visual row.

    Parsers disagree about what a "line" is: given a table row with wide gaps
    between columns, MuPDF emits one line *per cell* while pdfplumber emits one
    line for the whole row. Neither is wrong, but comparing them requires
    agreeing on the unit — and for row-structured documents the row is it. So
    fragments whose vertical extents overlap by more than ``overlap`` are
    merged, left to right.

    This is applied uniformly to every parser, so it normalizes the comparison
    rather than favouring one library's grouping.
    """
    boxed = [line for line in lines if line.bbox]
    if not boxed:
        return lines

    boxed.sort(key=lambda line: (line.bbox.y0, line.bbox.x0))
    rows: list[list[TextLine]] = []
    for line in boxed:
        placed = False
        if rows:
            current = rows[-1]
            top = min(f.bbox.y0 for f in current)
            bottom = max(f.bbox.y1 for f in current)
            height = max(bottom - top, 1e-6)
            span = min(bottom, line.bbox.y1) - max(top, line.bbox.y0)
            if span / min(height, max(line.bbox.height, 1e-6)) >= overlap:
                current.append(line)
                placed = True
        if not placed:
            rows.append([line])

    merged: list[TextLine] = []
    for group in rows:
        group.sort(key=lambda f: f.bbox.x0)
        tokens = [t for f in group for t in f.tokens]
        tokens.sort(key=lambda t: t[1].x0)
        merged.append(
            TextLine(
                page=group[0].page,
                text=" ".join(f.text for f in group if f.text).strip(),
                bbox=_merge([f.bbox for f in group]),
                tokens=tokens,
                source=group[0].source + ("+row" if len(group) > 1 else ""),
            )
        )
    unboxed = [line for line in lines if not line.bbox]
    return merged + unboxed


def reconstruct_lines(page: PageResult, merge: bool = True) -> list[TextLine]:
    """Best-effort lines for one page, in reading order."""
    line_blocks = [b for b in page.blocks if b.layer == "line"]
    if line_blocks:
        ordered = sorted(
            line_blocks,
            key=lambda b: (b.bbox.y0 if b.bbox else 0.0, b.bbox.x0 if b.bbox else 0.0),
        )
        words_by_line = _tokens_for(page)
        lines = []
        for b in ordered:
            if not b.text.strip():
                continue
            tokens = words_by_line(b.bbox)
            if not tokens and b.bbox:
                # No word layer (e.g. PDFium): treat the fragment as its own
                # token so column-position metrics still have something to use.
                tokens = [(b.text.strip(), b.bbox)]
            lines.append(
                TextLine(
                    page=page.page_number,
                    text=b.text.strip(),
                    bbox=b.bbox,
                    tokens=tokens,
                    source="line",
                )
            )
        return merge_rows(lines) if merge else lines

    word_lines = _lines_from_words(page)
    if word_lines:
        return word_lines

    # Figures carry no readable text, so a page whose only boxed blocks are
    # figures (Mistral OCR returns image boxes and nothing else) must fall
    # through to the page's own text rather than reporting one bogus line.
    region_blocks = [
        b for b in page.blocks if b.layer in ("region", "block") and b.kind != "figure" and b.text.strip()
    ]
    if region_blocks:
        ordered = sorted(
            region_blocks,
            key=lambda b: (b.order if b.order is not None else 0, b.bbox.y0 if b.bbox else 0.0),
        )
        out: list[TextLine] = []
        for b in ordered:
            for piece in b.text.splitlines():
                if piece.strip():
                    out.append(
                        TextLine(page=page.page_number, text=piece.strip(), bbox=b.bbox, source="region")
                    )
        return out

    return [
        TextLine(page=page.page_number, text=line.strip(), source="text")
        for line in page.text.splitlines()
        if line.strip()
    ]


def _tokens_for(page: PageResult):
    words = [b for b in page.blocks if b.layer == "word" and b.bbox]

    def lookup(bbox: BBox | None) -> list[tuple[str, BBox]]:
        if bbox is None or not words:
            return []
        inside = [
            (w.text, w.bbox)
            for w in words
            if w.bbox.intersection_area(bbox) > 0.5 * max(w.bbox.area, 1e-6)
        ]
        inside.sort(key=lambda t: t[1].x0)
        return inside

    return lookup


def all_lines(pages: list[PageResult]) -> list[TextLine]:
    return [line for page in pages for line in reconstruct_lines(page)]
