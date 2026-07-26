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


def is_available(cls: Type["PdfParser"]) -> bool:
    """Availability, treating a probe that blows up as "not available".

    Probes touch the outside world — importing a half-installed package, asking
    an ONNX runtime whether it loaded, shelling out to look for a binary — so
    they can raise anything at all. One adapter's bad day must not take the
    others down with it.
    """
    try:
        return cls.check_availability().available
    except Exception:
        return False


def available_parsers() -> list[Type["PdfParser"]]:
    return [c for c in all_parsers() if is_available(c)]


def describe_all() -> list[dict]:
    """Describe every parser, degrading a broken one into an unavailable entry.

    This is what the viewer's sidebar is built from, so it has to keep its
    promise: a list of every parser, always. Raising here would empty the
    sidebar and leave the UI with nothing to say beyond "failed to fetch".
    """
    out: list[dict] = []
    for cls in all_parsers():
        try:
            out.append(cls.describe())
        except Exception as exc:
            out.append(_broken(cls, exc))
    return out


def _broken(cls: Type["PdfParser"], exc: Exception) -> dict:
    return {
        "id": cls.id,
        "name": cls.name or cls.id,
        "kind": cls.kind,
        "description": cls.description,
        "homepage": cls.homepage,
        "tags": list(cls.tags),
        "extra": cls.extra,
        "cost_hint": cls.cost_hint,
        "options": [],
        "available": False,
        "unavailable_reason": f"{type(exc).__name__}: {exc}"[:300],
        "version": "",
    }


def __iter__() -> Iterator[Type["PdfParser"]]:  # pragma: no cover - convenience
    return iter(all_parsers())
