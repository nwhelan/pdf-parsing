"""Normalized document model shared by every parser adapter.

Coordinate convention
---------------------
Every bounding box in this project is expressed in **PDF points with a
top-left origin and y growing downwards**, matching how the page is rendered
to an image. Parsers that natively use a bottom-left origin (pdfminer,
pypdfium2, docling) convert on the way in via :meth:`BBox.from_bottom_left`.

A page also carries its ``width``/``height`` in points, so the viewer can map
points to rendered pixels with a single scale factor.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

# Layers are granularity bands. A parser may emit several of them for the same
# page; the viewer lets you toggle between them.
LAYER_WORD = "word"
LAYER_LINE = "line"
LAYER_BLOCK = "block"
LAYER_TABLE = "table"
LAYER_CELL = "cell"
LAYER_REGION = "region"

ALL_LAYERS = (LAYER_WORD, LAYER_LINE, LAYER_BLOCK, LAYER_REGION, LAYER_TABLE, LAYER_CELL)


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def from_bottom_left(cls, x0: float, y0: float, x1: float, y1: float, page_height: float) -> "BBox":
        """Convert a PDF-native (bottom-left origin) box to the top-left convention."""
        return cls(x0=x0, y0=page_height - y1, x1=x1, y1=page_height - y0).normalized()

    def normalized(self) -> "BBox":
        return BBox(
            x0=min(self.x0, self.x1),
            y0=min(self.y0, self.y1),
            x1=max(self.x0, self.x1),
            y1=max(self.y0, self.y1),
        )

    def scaled(self, factor: float) -> "BBox":
        return BBox(x0=self.x0 * factor, y0=self.y0 * factor, x1=self.x1 * factor, y1=self.y1 * factor)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def intersection_area(self, other: "BBox") -> float:
        dx = min(self.x1, other.x1) - max(self.x0, other.x0)
        dy = min(self.y1, other.y1) - max(self.y0, other.y0)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    def iou(self, other: "BBox") -> float:
        inter = self.intersection_area(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def is_finite(self) -> bool:
        return all(math.isfinite(v) for v in (self.x0, self.y0, self.x1, self.y1))


class Block(BaseModel):
    """A single piece of extracted content with an optional location."""

    id: str
    page: int  # 1-based
    layer: str = LAYER_BLOCK
    kind: str = "text"  # text | title | header | footer | list | figure | table | key_value | caption | formula
    text: str = ""
    bbox: BBox | None = None
    order: int | None = None
    confidence: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class TableCell(BaseModel):
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: BBox | None = None
    is_header: bool = False


class Table(BaseModel):
    id: str
    page: int
    bbox: BBox | None = None
    n_rows: int = 0
    n_cols: int = 0
    cells: list[TableCell] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def grid(self) -> list[list[str]]:
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.n_rows and 0 <= cell.col < self.n_cols:
                grid[cell.row][cell.col] = cell.text
        return grid

    def to_markdown(self) -> str:
        grid = self.grid()
        if not grid:
            return ""
        out = ["| " + " | ".join(c.replace("\n", " ") for c in grid[0]) + " |"]
        out.append("| " + " | ".join("---" for _ in grid[0]) + " |")
        for row in grid[1:]:
            out.append("| " + " | ".join(c.replace("\n", " ") for c in row) + " |")
        return "\n".join(out)


class PageResult(BaseModel):
    page_number: int  # 1-based
    width: float  # points
    height: float  # points
    rotation: int = 0
    blocks: list[Block] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)
    text: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)

    def blocks_in(self, layer: str) -> list[Block]:
        return [b for b in self.blocks if b.layer == layer]

    def layers(self) -> list[str]:
        seen: list[str] = []
        for b in self.blocks:
            if b.layer not in seen:
                seen.append(b.layer)
        return seen


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    requests: int = 0


class ParseResult(BaseModel):
    """The output of running one parser over one document."""

    parser_id: str
    parser_name: str = ""
    parser_version: str = ""
    doc_id: str = ""
    doc_name: str = ""
    status: str = "ok"  # ok | error
    error: str | None = None
    duration_s: float = 0.0
    per_page_s: dict[int, float] = Field(default_factory=dict)
    created_at: str = ""
    options: dict[str, Any] = Field(default_factory=dict)
    pages: list[PageResult] = Field(default_factory=list)
    markdown: str | None = None
    extraction: Any | None = None
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] = Field(default_factory=list)

    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def all_blocks(self) -> list[Block]:
        return [b for p in self.pages for b in p.blocks]

    def all_tables(self) -> list[Table]:
        return [t for p in self.pages for t in p.tables]

    def page(self, number: int) -> PageResult | None:
        for p in self.pages:
            if p.page_number == number:
                return p
        return None

    def layers(self) -> list[str]:
        seen: list[str] = []
        for p in self.pages:
            for layer in p.layers():
                if layer not in seen:
                    seen.append(layer)
        return [layer for layer in ALL_LAYERS if layer in seen] + [
            layer for layer in seen if layer not in ALL_LAYERS
        ]


class ParsedDocument(BaseModel):
    """What a parser adapter returns; the runner wraps it into a ParseResult."""

    pages: list[PageResult] = Field(default_factory=list)
    markdown: str | None = None
    # Structured output from a data-extraction request: whatever the parser was
    # asked for by schema, rather than the page as text. Comparable across
    # parsers precisely because the schema is the same for all of them.
    extraction: Any | None = None
    usage: Usage = Field(default_factory=Usage)
    warnings: list[str] = Field(default_factory=list)
    per_page_s: dict[int, float] = Field(default_factory=dict)
