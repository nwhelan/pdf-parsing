"""Parser adapters.

Importing this package registers every adapter. Adapters whose dependencies
are missing still register — they just report themselves as unavailable, so
the UI can show what you'd get by installing them.
"""

from __future__ import annotations

from ..registry import register
from .anthropic_parser import AnthropicVisionParser
from .base import Availability, Option, PdfParser
from .docling_parser import DoclingParser
from .gemini_parser import GeminiVisionParser
from .openai_parser import OpenAIVisionParser
from .pdfminer_parser import PdfminerParser
from .pdfplumber_parser import PdfplumberParser
from .pymupdf_parser import PyMuPDFParser
from .pypdfium2_parser import Pypdfium2Parser
from .tesseract_parser import TesseractParser
from .unstructured_parser import UnstructuredParser

for _cls in (
    PyMuPDFParser,
    PdfplumberParser,
    Pypdfium2Parser,
    PdfminerParser,
    DoclingParser,
    UnstructuredParser,
    TesseractParser,
    AnthropicVisionParser,
    OpenAIVisionParser,
    GeminiVisionParser,
):
    register(_cls)

__all__ = ["PdfParser", "Option", "Availability"]
