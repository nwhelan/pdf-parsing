"""Base class + availability probing for parser adapters."""

from __future__ import annotations

import importlib.util
import os
import re
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


# Anything that looks like a credential, wherever it turns up in a header or a
# request body. Debug output is written to disk, rendered in a browser and
# copied to a clipboard, so it has to be safe to hand to someone else.
#
# Matched as whole names or as a suffix, not as a substring: `token_param` and
# `max_completion_tokens` are parameter names worth reading, and a redactor that
# hides them makes the log less useful without making it safer.
SECRET_NAMES = frozenset(
    {
        "authorization",
        "api_key",
        "apikey",
        "x_api_key",
        "key",
        "token",
        "secret",
        "password",
        "credentials",
        "bearer",
    }
)
SECRET_SUFFIXES = ("_key", "_token", "_secret", "_password", "_credentials")

# Credentials also travel *inside* values: gateways and Azure accept a key as a
# query parameter, and a URL can carry basic-auth. Keys alone would miss those.
SECRET_QUERY_PARAMS = ("key", "token", "secret", "password", "sig", "signature", "code")
_QUERY_SECRET = re.compile(
    r"([?&](?:[\w-]*(?:" + "|".join(SECRET_QUERY_PARAMS) + r"))=)([^&\s]+)", re.IGNORECASE
)
_BASIC_AUTH = re.compile(r"(://)([^/@\s:]+):([^/@\s]+)@")
_BEARER = re.compile(r"\b(bearer\s+)(\S+)", re.IGNORECASE)

# Long values are almost always a base64 document or an image; the shape is the
# useful part, not the megabyte.
MAX_DEBUG_VALUE = 600

MASK = "<redacted>"


def is_secret(key: str) -> bool:
    name = key.lower().replace("-", "_")
    return name in SECRET_NAMES or name.endswith(SECRET_SUFFIXES)


def scrub(text: str) -> str:
    """Mask credentials carried inside a string, not just under a telling key."""
    text = _QUERY_SECRET.sub(lambda m: m.group(1) + MASK, text)
    text = _BASIC_AUTH.sub(lambda m: f"{m.group(1)}{m.group(2)}:{MASK}@", text)
    return _BEARER.sub(lambda m: m.group(1) + MASK, text)


def redact(value: Any, key: str = "") -> Any:
    """Copy a request structure so it is safe to store, show and share.

    Three jobs, and the third is not about secrecy: whatever comes back must be
    JSON-serializable, because this ends up inside a stored result. A value that
    cannot be serialized would otherwise fail the *save* — losing a result whose
    API call has already been paid for — so anything exotic becomes its repr.
    """
    if is_secret(key) and isinstance(value, str) and value:
        return f"<{len(value)} chars redacted>"
    if isinstance(value, dict):
        return {str(k): redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [redact(v, key) for v in value]
    if isinstance(value, str):
        value = scrub(value)
        if len(value) > MAX_DEBUG_VALUE:
            return f"{value[:MAX_DEBUG_VALUE]}… (+{len(value) - MAX_DEBUG_VALUE} chars)"
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    # Bytes, SDK objects, dates: keep something readable rather than risking the
    # whole result on a serializer that has never seen it.
    return redact(f"<{type(value).__name__}: {value!r}>"[:MAX_DEBUG_VALUE])


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

    # -- debug -----------------------------------------------------------

    @property
    def debug_events(self) -> list[dict[str, Any]]:
        """What this adapter sent, in order.

        The runner reads this whether :meth:`parse` returned or raised, which is
        the point: the request that *preceded* a failure is the thing you need
        to see, and by the time the error surfaces it is otherwise gone.
        """
        if not hasattr(self, "_debug_events"):
            self._debug_events: list[dict[str, Any]] = []
        return self._debug_events

    def record_request(self, label: str, **fields: Any) -> None:
        """Record one outgoing request, redacted and truncated."""
        self.debug_events.append({"event": label, **{k: redact(v, k) for k, v in fields.items()}})

    @property
    def parse_warnings(self) -> list[str]:
        """Notes for the user that don't stop the run — drained into the result."""
        if not hasattr(self, "_parse_warnings"):
            self._parse_warnings: list[str] = []
        return self._parse_warnings

    def note_warning(self, text: str) -> None:
        if text not in self.parse_warnings:
            self.parse_warnings.append(text)

    def record_response(self, label: str, body: Any, verbose: bool = False) -> None:
        """Record what came back. Bodies are kept only in verbose (debug) mode."""
        event: dict[str, Any] = {"event": label}
        if isinstance(body, dict):
            event["keys"] = sorted(body)
        if verbose:
            event["body"] = redact(body)
        self.debug_events.append(event)

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
    """Normalize a 1-based page selection against a document's page count.

    A selection that matches nothing is a mistake worth stopping for: it used to
    mean a remote parser posted an empty page list and paid for a request that
    could only come back empty, or made no request at all and reported a
    successful run of nothing.
    """
    if not pages:
        return list(range(1, total + 1))
    wanted = [p for p in pages if 1 <= p <= total]
    if not wanted:
        raise RuntimeError(
            f"no such page: {sorted(set(pages))} requested, but the document has "
            f"{total} page{'s' if total != 1 else ''}"
        )
    return wanted
