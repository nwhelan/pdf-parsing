"""Parser registry.

Adapters register themselves by subclassing :class:`PdfParser` and calling
:func:`register`. ``pdfplay.parsers`` imports every adapter module so that
importing the package is enough to populate the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Type

if TYPE_CHECKING:  # avoid a cycle: pdfplay.parsers imports this module
    from .parsers.base import PdfParser

_REGISTRY: dict[str, Type["PdfParser"]] = {}


def register(cls: Type["PdfParser"]) -> Type["PdfParser"]:
    if not cls.id:
        raise ValueError(f"{cls.__name__} must define an id")
    if cls.id in _REGISTRY and _REGISTRY[cls.id] is not cls:
        raise ValueError(f"duplicate parser id: {cls.id}")
    _REGISTRY[cls.id] = cls
    return cls


def get(parser_id: str) -> Type["PdfParser"]:
    try:
        return _REGISTRY[parser_id]
    except KeyError:
        raise KeyError(f"unknown parser: {parser_id!r} (known: {', '.join(sorted(_REGISTRY))})") from None


def all_parsers() -> list[Type["PdfParser"]]:
    return sorted(_REGISTRY.values(), key=lambda c: (c.kind != "local", c.id))


def available_parsers() -> list[Type["PdfParser"]]:
    return [c for c in all_parsers() if c.check_availability().available]


def describe_all() -> list[dict]:
    return [c.describe() for c in all_parsers()]


def __iter__() -> Iterator[Type["PdfParser"]]:  # pragma: no cover - convenience
    return iter(all_parsers())
