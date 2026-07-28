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

Two transports do the actual call:

``litellm``
    :func:`litellm.ocr`, which speaks the same request shape and already knows
    how to reach ``mistral``, ``azure_ai`` and ``vertex_ai``. It also resolves
    per-page pricing from its model table, so cost comes back measured rather
    than estimated. Preferred when the package is installed.
``http``
    A direct ``httpx`` POST using the URL and auth resolved here. Keeps the
    parser usable without litellm, and is the escape hatch for a gateway
    litellm doesn't model.

``transport=auto`` picks litellm when it's importable and falls back to HTTP.
Both return the same JSON shape, so everything downstream is shared.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

from ..geometry import page_geometry
from ..markdown_tables import tables_from_markdown
from ..models import BBox, Block, PageResult, ParsedDocument, Usage
from .base import Availability, Option, PdfParser, _importable, select_pages
from .vision_base import VisionParser

MISTRAL_URL = "https://api.mistral.ai/v1/ocr"

# Checked in order; the first one set wins unless `api_key_env` names another.
# `AZURE_AI_API_KEY` is litellm's own name for a Foundry key, so a workspace
# already set up for litellm needs no extra configuration here.
KEY_ENV_CANDIDATES = (
    "MISTRAL_API_KEY",
    "AZURE_AI_API_KEY",
    "AZURE_MISTRAL_API_KEY",
    "MISTRAL_OCR_API_KEY",
)
# Endpoint overrides, so a deployment URL doesn't have to live in the UI.
URL_ENV_CANDIDATES = ("MISTRAL_OCR_URL", "AZURE_MISTRAL_ENDPOINT", "AZURE_AI_API_BASE")

# The OCR payload is Mistral's shape wherever it is served, so a custom gateway
# is still routed through litellm's mistral provider — only the base URL moves.
LITELLM_PROVIDERS = {"mistral": "mistral", "azure": "azure_ai", "custom": "mistral"}


class MistralOCRParser(PdfParser):
    """Shared implementation; subclasses only set an id, a name and a model."""

    kind = "remote"
    model_default = "mistral-ocr-latest"
    homepage = "https://docs.mistral.ai/capabilities/OCR/basic_ocr/"
    tags = ("ocr", "remote", "markdown", "tables")
    requires = ("httpx",)
    extra = "mistral"
    cost_hint = "billed per page; $1-$4 per 1000 pages depending on the model"
    options = (
        Option(
            "model",
            "str",
            "",
            help=(
                "Model id, or an Azure deployment name (Foundry calls it "
                "mistral-document-ai-2505/-2512). Blank uses this parser's default."
            ),
        ),
        Option(
            "endpoint",
            "choice",
            "mistral",
            choices=["mistral", "azure", "custom"],
            help="Which service to call. 'azure' targets an Azure AI Foundry deployment.",
        ),
        Option(
            "transport",
            "choice",
            "auto",
            choices=["auto", "litellm", "http"],
            help="How to make the call. 'auto' uses litellm when installed, else a direct POST.",
        ),
        Option(
            "base_url",
            "str",
            "",
            help=(
                "The OCR endpoint; under litellm the base, whose path is completed for you. "
                f"Blank falls back to ${', $'.join(URL_ENV_CANDIDATES)}, then to the preset."
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
            help="HTTP transport only: 'auto' sends Bearer to Mistral and api-key to Azure.",
        ),
        Option(
            "api_version",
            "str",
            "",
            help="HTTP transport, Azure only: value for the ?api-version= query parameter.",
        ),
        Option("include_images", "bool", False, help="Ask for image crops as base64 (large responses)."),
        Option(
            "document_annotation_schema",
            "text",
            "",
            help=(
                "JSON Schema for document-level extraction. Mistral returns an object matching "
                "it alongside the Markdown. Paste a bare schema; the json_schema envelope is "
                "added for you."
            ),
        ),
        Option(
            "document_annotation_prompt",
            "text",
            "",
            help="Instructions to go with the extraction schema.",
        ),
        Option(
            "bbox_annotation_schema",
            "text",
            "",
            help="JSON Schema applied to each image region — captions, chart summaries, labels.",
        ),
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
    def configured_base(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> str:
        """The URL as configured, before any path or query is added to it.

        litellm wants the base and completes the path itself; the HTTP
        transport wants the completed URL. Both start here.
        """
        env = os.environ if env is None else env
        explicit = (opts.get("base_url") or "").strip()
        if explicit:
            return explicit
        return next((v for name in URL_ENV_CANDIDATES if (v := (env.get(name) or "").strip())), "")

    @classmethod
    def resolve_endpoint(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> str:
        """Work out which URL to POST to, most specific source first."""
        endpoint = opts.get("endpoint") or "mistral"
        base = cls.configured_base(opts, env)
        if base:
            # An explicit base_url is taken literally; an env var may be a bare
            # Foundry resource root, which still needs the OCR path.
            if (opts.get("base_url") or "").strip():
                return cls._with_api_version(base, opts)
            return cls._with_api_version(cls._normalize_azure(base, endpoint), opts)

        if endpoint == "mistral":
            return MISTRAL_URL
        raise RuntimeError(
            f"endpoint={endpoint!r} needs a URL: set the base_url option or "
            f"one of {', '.join(URL_ENV_CANDIDATES)}"
        )

    @classmethod
    def resolve_litellm_model(cls, opts: dict[str, Any], model: str) -> str:
        """Prefix the model with the litellm provider for the chosen endpoint.

        A model that already names a provider (``azure_ai/...``, ``vertex_ai/...``)
        is passed through untouched, which is how you reach a provider this
        adapter has no preset for.
        """
        if "/" in model:
            return model
        provider = LITELLM_PROVIDERS.get(opts.get("endpoint") or "mistral", "mistral")
        return f"{provider}/{model}"

    @classmethod
    def resolve_key(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> str:
        env = os.environ if env is None else env
        named = (opts.get("api_key_env") or "").strip()
        candidates = (named,) if named else KEY_ENV_CANDIDATES
        key = next((env[name] for name in candidates if env.get(name)), None)
        if not key:
            raise RuntimeError(f"no API key: set {' or '.join(candidates)}")
        return key

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
        key = cls.resolve_key(opts, env)
        scheme = opts.get("auth_header") or "auto"
        if scheme == "auto":
            scheme = "api-key" if (opts.get("endpoint") == "azure") else "bearer"
        if scheme == "api-key":
            return {"api-key": key}
        return {"Authorization": f"Bearer {key}"}

    # -- data extraction -------------------------------------------------

    @classmethod
    def annotation_params(cls, opts: dict[str, Any]) -> dict[str, Any]:
        """Mistral's structured-extraction parameters, from the schema options.

        The API wants each schema wrapped in a ``json_schema`` envelope. Asking
        for that by hand is a papercut in a text box, so a bare schema is
        wrapped here and an already-wrapped one is passed through.
        """
        params: dict[str, Any] = {}
        for option, param, name in (
            ("document_annotation_schema", "document_annotation_format", "document_annotation"),
            ("bbox_annotation_schema", "bbox_annotation_format", "bbox_annotation"),
        ):
            schema = VisionParser.parse_schema(opts.get(option), label=option)
            if schema is not None:
                params[param] = _as_json_schema(schema, name)

        prompt = str(opts.get("document_annotation_prompt") or "").strip()
        if prompt:
            params["document_annotation_prompt"] = prompt
        return params

    # -- transports (patched in tests) -----------------------------------

    def use_litellm(self, opts: dict[str, Any]) -> bool:
        transport = opts.get("transport") or "auto"
        if transport == "auto":
            return _importable("litellm")
        return transport == "litellm"

    def post_ocr(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict:
        import httpx

        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"{url} returned {response.status_code}: {response.text[:400]}")
        return response.json()

    def ocr_via_litellm(self, opts: dict[str, Any], model: str, payload: dict[str, Any]) -> dict:
        """Same request, routed by litellm.

        The response is normalized to Mistral's OCR shape by litellm itself, so
        it maps exactly like the raw one. Its measured cost, when the model is
        in litellm's price table, rides along in ``_cost_usd``.
        """
        import litellm

        extras = {k: v for k, v in payload.items() if k not in ("model", "document")}
        response = litellm.ocr(
            model=self.resolve_litellm_model(opts, model),
            document=payload["document"],
            api_key=self.resolve_key(opts),
            api_base=self.configured_base(opts) or None,
            timeout=float(opts["timeout_s"]),
            **extras,
        )
        body = response.model_dump()
        cost = getattr(response, "_hidden_params", {}).get("response_cost")
        if cost is not None:
            body["_cost_usd"] = cost
        return body

    # -- driver ----------------------------------------------------------

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        opts = self.resolved_options(options)
        geometry = page_geometry(pdf_path)
        wanted = select_pages(len(geometry), pages)
        model = (opts.get("model") or "").strip() or self.model_default

        data_url = "data:application/pdf;base64," + base64.standard_b64encode(
            pdf_path.read_bytes()
        ).decode("ascii")
        payload: dict[str, Any] = {
            "model": model,
            "document": {"type": "document_url", "document_url": data_url},
            "include_image_base64": bool(opts["include_images"]),
            **self.annotation_params(opts),
        }
        if pages:
            payload["pages"] = [p - 1 for p in wanted]  # the API counts pages from 0

        started = time.perf_counter()
        if self.use_litellm(opts):
            body = self.ocr_via_litellm(opts, model, payload)
        else:
            body = self.post_ocr(
                self.resolve_endpoint(opts),
                {"Content-Type": "application/json", **self.resolve_auth(opts)},
                payload,
                float(opts["timeout_s"]),
            )
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
            if raw.get("image_annotations"):  # bbox_annotation_format results
                page.meta["image_annotations"] = raw["image_annotations"]
            out_pages.append(page)

        if not out_pages:
            warnings.append("the response contained no pages")

        pages_processed = int((body.get("usage_info") or {}).get("pages_processed") or len(out_pages))
        # litellm knows the real per-page price for the models in its table;
        # the option is the estimate used when nothing measured it.
        measured = body.get("_cost_usd")
        cost = (
            float(measured)
            if measured is not None
            else pages_processed * float(opts["price_per_1k_pages"]) / 1000.0
        )
        usage = Usage(model=body.get("model") or model, requests=1, cost_usd=round(cost, 6))

        # The annotation comes back as a JSON *string*, since the schema was
        # sent as a response format. Decode it so it lands next to every other
        # parser's extraction rather than as an opaque blob.
        annotation = body.get("document_annotation")
        if isinstance(annotation, str) and annotation.strip():
            try:
                annotation = json.loads(annotation)
            except json.JSONDecodeError:
                warnings.append("document_annotation was not valid JSON; kept as text")

        per_page = {p.page_number: elapsed / max(1, len(out_pages)) for p in out_pages}
        return ParsedDocument(
            pages=out_pages,
            markdown="\n\n---\n\n".join(markdown_parts) if markdown_parts else None,
            extraction=annotation or None,
            usage=usage,
            warnings=warnings,
            per_page_s=per_page,
        )


def _as_json_schema(schema: dict[str, Any], name: str) -> dict[str, Any]:
    if schema.get("type") == "json_schema" and "json_schema" in schema:
        return schema  # already an envelope
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


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
    # Mistral publishes dated ids rather than generation numbers; 2512 is the
    # release before the 4 series. Override `model` for a different one.
    model_default = "mistral-ocr-2512"
    description = (
        "Mistral OCR (mistral-ocr-2512) over the document-level OCR API. Returns Markdown per "
        "page (tables included) and image boxes, but no text geometry. Point it at "
        "api.mistral.ai, an Azure AI Foundry deployment, or any gateway via the endpoint "
        "options. On Foundry the deployment is named mistral-document-ai-2512."
    )


class MistralOCR4Parser(MistralOCRParser):
    id = "mistral-ocr-4"
    name = "Mistral OCR 4"
    model_default = "mistral-ocr-4-0"
    description = (
        "Mistral OCR 4 (mistral-ocr-4-0) over the same API, registered separately so the two "
        "generations can be run side by side on the same document — including against "
        "different endpoints, since options are part of the result cache key."
    )
