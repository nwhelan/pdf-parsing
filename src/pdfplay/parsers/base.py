"""Base class + availability probing for parser adapters."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ParsedDocument


def _importable(module: str) -> bool:
    """find_spec, but tolerant of a missing parent package (e.g. `google.genai`)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class Availability:
    available: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason}


class Option:
    """Declarative option, surfaced as a control in the web UI."""

    def __init__(
        self,
        name: str,
        type: str,
        default: Any,
        label: str = "",
        choices: list[Any] | None = None,
        help: str = "",
    ) -> None:
        self.name = name
        self.type = type  # bool | int | float | str | choice
        self.default = default
        self.label = label or name.replace("_", " ").title()
        self.choices = choices
        self.help = help

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "label": self.label,
            "choices": self.choices,
            "help": self.help,
        }


class PdfParser:
    """Adapter interface. Subclasses implement :meth:`parse`.

    Adapters never handle timing, caching, or error capture — the runner does.
    """

    id: str = ""
    name: str = ""
    kind: str = "local"  # local | remote
    description: str = ""
    homepage: str = ""
    tags: tuple[str, ...] = ()
    # Python modules that must be importable
    requires: tuple[str, ...] = ()
    # Environment variables that must be set (remote parsers)
    env_vars: tuple[str, ...] = ()
    # Extra needed to install it, e.g. `pip install -e '.[docling]'`
    extra: str = ""
    options: tuple[Option, ...] = ()
    cost_hint: str = "free"

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------

    @classmethod
    def check_availability(cls) -> Availability:
        missing = [m for m in cls.requires if not _importable(m)]
        if missing:
            hint = f" (pip install -e '.[{cls.extra}]')" if cls.extra else ""
            return Availability(False, f"missing module(s): {', '.join(missing)}{hint}")
        unset = [v for v in cls.env_vars if not os.environ.get(v)]
        if unset:
            return Availability(False, f"unset env var(s): {', '.join(unset)}")
        return Availability(True)

    @classmethod
    def version(cls) -> str:
        import importlib.metadata as md

        for module in cls.requires:
            try:
                return f"{module} {md.version(module)}"
            except Exception:
                pass
            try:
                mod = __import__(module)
                v = getattr(mod, "__version__", None)
                if v:
                    return f"{module} {v}"
            except Exception:  # pragma: no cover - best effort only
                continue
        return ""

    @classmethod
    def resolved_options(cls, options: dict[str, Any] | None) -> dict[str, Any]:
        merged = {opt.name: opt.default for opt in cls.options}
        merged.update(options or {})
        return merged

    @classmethod
    def describe(cls) -> dict[str, Any]:
        avail = cls.check_availability()
        return {
            "id": cls.id,
            "name": cls.name or cls.id,
            "kind": cls.kind,
            "description": cls.description,
            "homepage": cls.homepage,
            "tags": list(cls.tags),
            "extra": cls.extra,
            "cost_hint": cls.cost_hint,
            "options": [o.as_dict() for o in cls.options],
            "available": avail.available,
            "unavailable_reason": avail.reason,
            "version": cls.version() if avail.available else "",
        }


def select_pages(total: int, pages: list[int] | None) -> list[int]:
    """Normalize a 1-based page selection against a document's page count."""
    if not pages:
        return list(range(1, total + 1))
    return [p for p in pages if 1 <= p <= total]
