"""Recover table structure from GitHub-flavored Markdown.

Several parsers describe a table only as Markdown — pymupdf-layout classifies
the region and writes the cells into the page's Markdown, Mistral OCR returns
Markdown per page and nothing else. Parsing those tables back into the
normalized :class:`~pdfplay.models.Table` is what makes their tables comparable
with parsers that expose real cell geometry.
"""

from __future__ import annotations

import re
from typing import Any

from .models import BBox, Table, TableCell

_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def is_row(line: str) -> bool:
    return bool(_ROW.match(line)) and not _RULE.match(line)


def split_row(line: str) -> list[str]:
    """Split one Markdown row into cells, unwrapping `<br>` into newlines."""
    match = _ROW.match(line)
    if match is None:
        return []
    return [cell.strip().replace("<br>", "\n").strip() for cell in match.group(1).split("|")]


def parse_markdown_tables(text: str) -> list[list[list[str]]]:
    """Return every GFM table in ``text`` as a list of rows of cells."""
    lines = text.splitlines()
    tables: list[list[list[str]]] = []
    i = 0
    while i < len(lines):
        if not (_ROW.match(lines[i]) and i + 1 < len(lines) and _RULE.match(lines[i + 1])):
            i += 1
            continue
        rows = [split_row(lines[i])]
        i += 2
        while i < len(lines) and is_row(lines[i]):
            rows.append(split_row(lines[i]))
            i += 1
        tables.append(rows)
    return tables


def tables_from_markdown(
    page_number: int,
    text: str,
    boxes: list[BBox | None] | None = None,
    meta: dict[str, Any] | None = None,
    id_prefix: str = "md",
) -> list[Table]:
    """Build :class:`Table` objects from the Markdown tables on one page.

    ``boxes`` are the parser's own table regions, matched to the Markdown
    tables in document order when the parser reports them.
    """
    boxes = boxes or []
    out: list[Table] = []
    for index, rows in enumerate(parse_markdown_tables(text)):
        cells = [
            TableCell(row=r, col=c, text=value, is_header=(r == 0))
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
        ]
        out.append(
            Table(
                id=f"p{page_number}-{id_prefix}{index}",
                page=page_number,
                bbox=boxes[index] if index < len(boxes) else None,
                n_rows=len(rows),
                n_cols=max((len(r) for r in rows), default=0),
                cells=cells,
                meta={"source": "markdown", **(meta or {})},
            )
        )
    return out
