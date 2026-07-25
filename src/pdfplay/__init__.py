"""pdfplay — a playground for comparing PDF parsers on the same document."""

from .models import BBox, Block, PageResult, ParsedDocument, ParseResult, Table, TableCell
from .workspace import Workspace

__version__ = "0.1.0"
__all__ = [
    "BBox",
    "Block",
    "PageResult",
    "ParseResult",
    "ParsedDocument",
    "Table",
    "TableCell",
    "Workspace",
]
