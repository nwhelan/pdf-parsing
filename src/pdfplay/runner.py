"""Runs parsers over documents: timing, error capture, and result caching."""

from __future__ import annotations

import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Iterable

from . import parsers as _parsers  # noqa: F401  (registers adapters)
from .models import ParseResult
from .registry import get as get_parser
from .workspace import Workspace, options_key


def run_parser(
    workspace: Workspace,
    doc_id: str,
    parser_id: str,
    pages: list[int] | None = None,
    options: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[ParseResult, str, bool]:
    """Run one parser over one document.

    Returns ``(result, cache_key, from_cache)``. Adapter failures are captured
    into a ``status="error"`` result rather than raised, so one broken parser
    never takes down a comparison run.
    """
    parser_cls = get_parser(parser_id)
    resolved = parser_cls.resolved_options(options)
    key = options_key(parser_id, resolved, pages)

    if not force:
        cached = workspace.load_result(doc_id, key)
        # Only successes are worth replaying: an error's cause is usually
        # outside the options hash (env var, config.yaml, the endpoint itself),
        # so a rerun must actually retry.
        if cached is not None and cached.status == "ok":
            return cached, key, True

    meta = workspace.get_document(doc_id)
    pdf_path = workspace.pdf_path(doc_id)

    result = ParseResult(
        parser_id=parser_id,
        parser_name=parser_cls.name or parser_id,
        parser_version=parser_cls.version(),
        doc_id=doc_id,
        doc_name=meta.name,
        options=resolved,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    availability = parser_cls.check_availability()
    if not availability.available:
        result.status = "error"
        result.error = f"parser unavailable: {availability.reason}"
        try:
            workspace.save_result(result, key)
        except Exception:  # pragma: no cover - nothing useful to cache anyway
            pass
        return result, key, False

    parser = parser_cls()
    started = time.perf_counter()
    try:
        parsed = parser.parse(pdf_path, pages, resolved)
        result.pages = parsed.pages
        result.markdown = parsed.markdown
        result.extraction = parsed.extraction
        result.usage = parsed.usage
        result.warnings = parsed.warnings
        result.per_page_s = parsed.per_page_s
        result.debug = parsed.debug or parser.debug_events
    except Exception as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.warnings.append(traceback.format_exc(limit=6))
        # The request that preceded the failure is the thing worth seeing, and
        # it only exists on the adapter instance — hence keeping one around.
        result.debug = parser.debug_events
    result.duration_s = time.perf_counter() - started

    # Caching is a convenience; the result is the thing that was paid for. A
    # full disk or an unserializable field must not turn a finished run into
    # nothing — record why it wasn't cached and hand it back anyway.
    try:
        workspace.save_result(result, key)
    except Exception as exc:
        result.warnings.append(f"result could not be cached: {type(exc).__name__}: {exc}")
    return result, key, False


def run_many(
    workspace: Workspace,
    doc_id: str,
    parser_ids: Iterable[str],
    pages: list[int] | None = None,
    options: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
    max_workers: int = 4,
) -> list[tuple[ParseResult, str, bool]]:
    """Run several parsers over the same document.

    Local parsers are CPU-bound and remote ones are I/O-bound; a small thread
    pool is a decent compromise that keeps remote calls overlapping.
    """
    parser_ids = list(parser_ids)
    per_parser = options or {}

    def one(pid: str) -> tuple[ParseResult, str, bool]:
        return run_parser(workspace, doc_id, pid, pages, per_parser.get(pid), force)

    if len(parser_ids) == 1 or max_workers <= 1:
        return [one(pid) for pid in parser_ids]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(parser_ids))) as pool:
        return list(pool.map(one, parser_ids))
