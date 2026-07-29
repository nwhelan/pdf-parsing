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

The call itself is a direct ``httpx`` POST. This endpoint is not part of the
OpenAI protocol — there is no ``/v1/ocr`` in the OpenAI API and no SDK client
that speaks it — so unlike the chat models here, it cannot go through the
OpenAI client and is spoken to directly.
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
from .base import Availability, Option, PdfParser, select_pages
from .vision_base import VisionParser

MISTRAL_URL = "https://api.mistral.ai/v1/ocr"

# Checked in order; the first one set wins unless `api_key_env` names another.
# `AZURE_AI_API_KEY` / `AZURE_AI_API_BASE` are the conventional Foundry names,
# so a workspace already configured for it needs nothing extra here.
KEY_ENV_CANDIDATES = (
    "MISTRAL_API_KEY",
    "AZURE_AI_API_KEY",
    "AZURE_MISTRAL_API_KEY",
    "MISTRAL_OCR_API_KEY",
)
# Endpoint overrides, so a deployment URL doesn't have to live in the UI.
URL_ENV_CANDIDATES = ("MISTRAL_OCR_URL", "AZURE_MISTRAL_ENDPOINT", "AZURE_AI_API_BASE")

# Foundry exposes Mistral OCR under two conventions depending on how it was
# deployed, and guessing the wrong one is a 404 that explains nothing:
#   serverless (models-as-a-service)  <host>.models.ai.azure.com/v1/ocr
#   an AI Services resource          <host>.services.ai.azure.com/providers/mistral/azure/ocr
SERVERLESS_HOST = ".models.ai.azure.com"
SERVERLESS_PATH = "/v1/ocr"
AI_SERVICES_PATH = "/providers/mistral/azure/ocr"


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
            "base_url",
            "str",
            "",
            help=(
                "Full URL of the OCR endpoint. An Azure resource root is completed for you. "
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
            help="'auto' sends Bearer to Mistral and api-key to Azure. Override for gateways.",
        ),
        Option(
            "api_version",
            "str",
            "",
            help="Azure only: value for the ?api-version= query parameter.",
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
        Option(
            "debug",
            "bool",
            False,
            help="Also record the raw response body in the result's debug log.",
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
        """The URL as configured, before any path or query is added to it."""
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
            # A URL that already has a path is taken literally — it may point at
            # a gateway with its own routing. A bare host is completed, whichever
            # box it came from: posting a document to a hostname is never what
            # was meant, and the resulting 404 explains nothing.
            return cls._with_api_version(cls._normalize_azure(base, endpoint), opts)

        if endpoint == "mistral":
            return MISTRAL_URL
        raise RuntimeError(
            f"endpoint={endpoint!r} needs a URL: set the base_url option or "
            f"one of {', '.join(URL_ENV_CANDIDATES)}"
        )

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
        """Complete the OCR path on a bare Foundry host.

        Which path depends on how the model was deployed, so it follows the
        hostname rather than assuming one convention.
        """
        url = url.rstrip("/")
        host = url.split("//")[-1].split("/")[0].lower()
        has_path = bool(url.split(host, 1)[-1].split("?")[0].strip("/"))
        if endpoint != "azure" or has_path:
            return url
        return f"{url}{SERVERLESS_PATH if host.endswith(SERVERLESS_HOST) else AI_SERVICES_PATH}"

    @classmethod
    def endpoint_hints(cls, url: str, opts: dict[str, Any], model: str) -> list[str]:
        """Configuration that will fail, named before the API refuses it."""
        hints: list[str] = []
        host = url.split("//")[-1].split("/")[0].lower()
        path = url.split(host, 1)[-1].split("?")[0]

        if host.endswith(".openai.azure.com"):
            hints.append(
                f"{host} is an Azure OpenAI resource, which does not serve Mistral OCR. On "
                f"Foundry the host is *{SERVERLESS_HOST} or *.services.ai.azure.com."
            )
        if not path.strip("/"):
            hints.append(
                f"The URL has no path, so this posts to the host root. Expected {SERVERLESS_PATH} "
                f"or {AI_SERVICES_PATH} — endpoint=azure appends one; endpoint=custom does not."
            )
        if opts.get("endpoint") == "azure" and model.startswith("mistral-ocr"):
            hints.append(
                f"{model!r} is an api.mistral.ai model id. On Foundry this field is the "
                "deployment name, usually mistral-document-ai-2505 or -2512."
            )
        if host.endswith(SERVERLESS_HOST) and not (opts.get("api_version") or "").strip():
            hints.append("A serverless Foundry endpoint usually requires api_version to be set.")
        return hints

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

        url = self.resolve_endpoint(opts)
        headers = {"Content-Type": "application/json", **self.resolve_auth(opts)}
        verbose = bool(opts.get("debug"))

        # Recorded before the POST: a 400 about a missing annotation prompt or a
        # 404 on a deployment name is only diagnosable against what was sent.
        # `wire` is the request and nothing else — an earlier version put the
        # `endpoint` option alongside the payload, and it read as a body field.
        hints = self.endpoint_hints(url, opts, model)
        self.record_request(
            "POST",
            wire={"url": url, "headers": headers, "body": payload},
            context={"endpoint_option": opts.get("endpoint"), "model": model},
            hints=hints,
        )

        started = time.perf_counter()
        body = self.post_ocr(url, headers, payload, float(opts["timeout_s"]))
        elapsed = time.perf_counter() - started
        self.record_response("ocr", body, verbose=verbose)

        warnings: list[str] = list(hints)
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
        cost = pages_processed * float(opts["price_per_1k_pages"]) / 1000.0
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
            debug=self.debug_events,
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
