"""Scoring: generic quality signals plus per-document-class scorers."""

from . import bank_statement, extraction, generic
from .lines import TextLine, reconstruct_lines

DOC_CLASSES = {
    "bank_statement": bank_statement,
}

__all__ = [
    "generic",
    "bank_statement",
    "extraction",
    "reconstruct_lines",
    "TextLine",
    "DOC_CLASSES",
]
