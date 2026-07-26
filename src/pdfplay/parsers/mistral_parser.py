"""Mistral OCR adapters.

Mistral OCR is a document-level API: you post the whole PDF and get Markdown
back per page, plus the bounding boxes of any *images* it found. It returns no
geometry for text, so the overlay for these parsers shows figures only — the
comparison value is in the Markdown, the recovered tables, and how well the
ledger survives.

Endpoints are configurable because the same model is served from several
places, each with its own URL shape and auth header:

``mistral``  ``https://api.mistral.ai/v1/ocr``, ``Authorization: Bearer``
``azure``    an Azure AI Foundry deployment, ``api-key`` header by default
``custom``   anything else (a gateway, a proxy, a self-hosted deployment)

Each registered parser keeps its own options, so one can point at Foundry while
another talks to api.mistral.ai in the same workspace, and their results are
cached separately.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from ..geometry import page_geometry
from ..markdown_tables import tables_from_markdown
from ..models import BBox, Block, PageResult, ParsedDocument, Usage
from .base import Availability, Option, PdfParser, select_pages

MISTRAL_URL = "https://api.mistral.ai/v1/ocr"

# Checked in order; the first one set wins unless `api_key_env` names another.
KEY_ENV_CANDIDATES = ("MISTRAL_API_KEY", "AZURE_MISTRAL_API_KEY", "MISTRAL_OCR_API_KEY")
# Endpoint overrides, so a deployment URL doesn't have to live in the UI.
URL_ENV_CANDIDATES = ("MISTRAL_OCR_URL", "AZURE_MISTRAL_ENDPOINT")


class MistralOCRParser(PdfParser):
    """Shared implementation; subclasses only set an id, a name and a model."""

    kind = "remote"
    model_default = "mistral-ocr-latest"
    homepage = "https://docs.mistral.ai/capabilities/OCR/basic_ocr/"
    tags = ("ocr", "remote", "markdown", "tables")
    requires = ("httpx",)
    extra = "mistral"
    cost_hint = "billed per page; ~$1 per 1000 pages on the Mistral API"
    options = (
        Option(
            "model",
            "str",
            "",
            help="Model id, or an Azure deployment name. Blank uses this parser's default.",
        ),
        Option(
            "endpoint",
            "choice",
            "mistral",
            choices=["mistral", "azure", "custom"],
            help="Which service to call. 'azure' targets an Azure AI Foundry deployment.",
        ),
        Option(
            "base_url",
            "str",
            "",
            help=(
                "Full URL of the OCR endpoint. Blank falls back to $MISTRAL_OCR_URL / "
                "$AZURE_MISTRAL_ENDPOINT, then to the preset for the chosen endpoint."
            ),
        ),
        Option(
            "api_key_env",
            "str",
            "",
            help=f"Env var holding the key. Blank tries {', '.join(KEY_ENV_CANDIDATES)}.",
        ),
        Option(
            "auth_header",
            "choice",
            "auto",
            choices=["auto", "bearer", "api-key"],
            help="'auto' sends Bearer to Mistral and api-key to Azure. Override for gateways.",
        ),
        Option("api_version", "str", "", help="Azure only: value for the ?api-version= query parameter."),
        Option("include_images", "bool", False, help="Ask for image crops as base64 (large responses)."),
        Option("timeout_s", "int", 300),
        Option("price_per_1k_pages", "float", 1.0, help="Used only to report an estimated cost."),
    )

    # -- availability ----------------------------------------------------

    @classmethod
    def check_availability(cls) -> Availability:
        base = super().check_availability()
        if not base.available:
            return base
        if not any(os.environ.get(name) for name in KEY_ENV_CANDIDATES):
            return Availability(False, f"set one of: {', '.join(KEY_ENV_CANDIDATES)}")
        return base

    # -- endpoint resolution ---------------------------------------------

    @classmethod
    def resolve_endpoint(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> str:
        """Work out which URL to call, most specific source first."""
        env = os.environ if env is None else env
        explicit = (opts.get("base_url") or "").strip()
        if explicit:
            return cls._with_api_version(explicit, opts)

        endpoint = opts.get("endpoint") or "mistral"
        for name in URL_ENV_CANDIDATES:
            value = (env.get(name) or "").strip()
            if value:
                return cls._with_api_version(cls._normalize_azure(value, endpoint), opts)

        if endpoint == "mistral":
            return MISTRAL_URL
        raise RuntimeError(
            f"endpoint={endpoint!r} needs a URL: set the base_url option or "
            f"one of {', '.join(URL_ENV_CANDIDATES)}"
        )

    @staticmethod
    def _normalize_azure(url: str, endpoint: str) -> str:
        """Accept a bare Foundry resource root and complete the OCR path."""
        url = url.rstrip("/")
        if endpoint != "azure" or "/ocr" in url:
            return url
        return f"{url}/providers/mistral/azure/ocr"

    @staticmethod
    def _with_api_version(url: str, opts: dict[str, Any]) -> str:
        version = (opts.get("api_version") or "").strip()
        if not version or "api-version=" in url:
            return url
        return f"{url}{'&' if '?' in url else '?'}api-version={version}"

    @classmethod
    def resolve_auth(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, str]:
        env = os.environ if env is None else env
        named = (opts.get("api_key_env") or "").strip()
        candidates = (named,) if named else KEY_ENV_CANDIDATES
        key = next((env[name] for name in candidates if env.get(name)), None)
        if not key:
            raise RuntimeError(f"no API key: set {' or '.join(candidates)}")

        scheme = opts.get("auth_header") or "auto"
        if scheme == "auto":
            scheme = "api-key" if (opts.get("endpoint") == "azure") else "bearer"
        if scheme == "api-key":
            return {"api-key": key}
        return {"Authorization": f"Bearer {key}"}

    # -- transport (patched in tests) ------------------------------------

    def post_ocr(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict:
        import httpx

        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"{url} returned {response.status_code}: {response.text[:400]}")
        return response.json()

    # -- driver ----------------------------------------------------------

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        opts = self.resolved_options(options)
        geometry = page_geometry(pdf_path)
        wanted = select_pages(len(geometry), pages)

        url = self.resolve_endpoint(opts)
        headers = {"Content-Type": "application/json", **self.resolve_auth(opts)}
        model = (opts.get("model") or "").strip() or self.model_default

        data_url = "data:application/pdf;base64," + base64.standard_b64encode(
            pdf_path.read_bytes()
        ).decode("ascii")
        payload: dict[str, Any] = {
            "model": model,
            "document": {"type": "document_url", "document_url": data_url},
            "include_image_base64": bool(opts["include_images"]),
        }
        if pages:
            payload["pages"] = [p - 1 for p in wanted]  # the API counts pages from 0

        started = time.perf_counter()
        body = self.post_ocr(url, headers, payload, float(opts["timeout_s"]))
        elapsed = time.perf_counter() - started

        warnings: list[str] = []
        out_pages: list[PageResult] = []
        markdown_parts: list[str] = []
        raw_pages = body.get("pages") or []

        for position, raw in enumerate(raw_pages):
            index = raw.get("index")
            page_no = int(index) + 1 if isinstance(index, int) else (
                wanted[position] if position < len(wanted) else position + 1
            )
            geo = geometry.get(page_no)
            if geo is None:
                warnings.append(f"response mentions page {page_no}, which the PDF does not have")
                continue

            text = raw.get("markdown") or ""
            markdown_parts.append(text)
            page = PageResult(
                page_number=page_no,
                width=geo.width,
                height=geo.height,
                rotation=geo.rotation,
                text=text,
                meta={"markdown": text, "model": body.get("model") or model},
            )
            _add_image_regions(page, raw)
            page.tables.extend(tables_from_markdown(page_no, text, meta={"model": model}))
            out_pages.append(page)

        if not out_pages:
            warnings.append("the response contained no pages")

        pages_processed = int((body.get("usage_info") or {}).get("pages_processed") or len(out_pages))
        usage = Usage(
            model=body.get("model") or model,
            requests=1,
            cost_usd=round(pages_processed * float(opts["price_per_1k_pages"]) / 1000.0, 6),
        )

        per_page = {p.page_number: elapsed / max(1, len(out_pages)) for p in out_pages}
        return ParsedDocument(
            pages=out_pages,
            markdown="\n\n---\n\n".join(markdown_parts) if markdown_parts else None,
            usage=usage,
            warnings=warnings,
            per_page_s=per_page,
        )


def _add_image_regions(page: PageResult, raw: dict[str, Any]) -> None:
    """Image boxes come back in pixels at the page's reported dimensions."""
    dims = raw.get("dimensions") or {}
    px_width = float(dims.get("width") or 0)
    px_height = float(dims.get("height") or 0)
    sx = page.width / px_width if px_width else 1.0
    sy = page.height / px_height if px_height else 1.0

    for i, image in enumerate(raw.get("images") or []):
        try:
            box = BBox(
                x0=float(image["top_left_x"]) * sx,
                y0=float(image["top_left_y"]) * sy,
                x1=float(image["bottom_right_x"]) * sx,
                y1=float(image["bottom_right_y"]) * sy,
            ).normalized()
        except (KeyError, TypeError, ValueError):
            continue
        page.blocks.append(
            Block(
                id=f"p{page.page_number}-img{i}",
                page=page.page_number,
                layer="region",
                kind="figure",
                bbox=box,
                order=i,
                meta={"image_id": image.get("id")},
            )
        )


class MistralOCR3Parser(MistralOCRParser):
    id = "mistral-ocr-3"
    name = "Mistral OCR 3"
    model_default = "mistral-ocr-3"
    description = (
        "Mistral OCR 3 over the document-level OCR API. Returns Markdown per page (tables "
        "included) and image boxes, but no text geometry. Point it at api.mistral.ai, an "
        "Azure AI Foundry deployment, or any gateway via the endpoint options."
    )


class MistralOCR4Parser(MistralOCRParser):
    id = "mistral-ocr-4"
    name = "Mistral OCR 4"
    model_default = "mistral-ocr-4"
    description = (
        "Mistral OCR 4 over the same API as OCR 3, registered separately so the two versions "
        "can be run side by side on the same document — including against different endpoints."
    )
