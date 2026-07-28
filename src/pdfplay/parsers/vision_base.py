"""Shared machinery for vision-model parsers.

Every vision adapter does the same three things: rasterize a page, ask a
multimodal model for a JSON transcription with normalized boxes, and map those
boxes back into PDF points. Only the API call differs, so subclasses implement
:meth:`call_model` and inherit the rest.

Box convention on the wire: integers in ``[0, 1000]`` relative to the *image*.
Most models are asked for ``[x0, y0, x1, y1]``; Gemini is asked for its native
``[ymin, xmin, ymax, xmax]`` and declares ``bbox_order = "yxyx"``.

Two options make these adapters usable for comparing *models* rather than
libraries. ``instructions`` edits the prompt — appended to the transcription
prompt, or replacing it outright — and ``extraction_schema`` takes a JSON Schema
that is added to the response as an ``extraction`` field. Because both are
ordinary options they are part of the cache key, so the same page under two
prompts is two comparable results rather than one overwriting the other.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any

from ..geometry import page_geometry, render_page_image
from ..models import BBox, Block, PageResult, ParsedDocument, Usage
from .base import Option, PdfParser, select_pages

KINDS = ["text", "title", "header", "footer", "list", "table", "key_value", "figure", "caption"]

PROMPT = """You are a document parsing engine. Transcribe this page image exactly.

Return every distinct text region on the page, in natural reading order. For each region:
- `text`: the verbatim text, preserving line breaks inside the region with \\n.
  Never summarize, correct, translate, or omit anything — including page numbers,
  headers, footers, fine print, and numbers in table cells.
- `kind`: one of {kinds}.
- `bbox`: {bbox_help}

For tables, emit one region per table row (kind "table"), with the row's cells
joined by " | " in reading order, so that column structure survives.

Also return `markdown`: the whole page as clean GitHub-flavored Markdown, with
tables as Markdown tables. Do not wrap it in a code fence."""

BBOX_HELP_XYXY = (
    "`[x0, y0, x1, y1]` as integers from 0 to 1000, relative to the image "
    "(x from the left edge, y from the top edge)."
)
BBOX_HELP_YXYX = (
    "`[ymin, xmin, ymax, xmax]` as integers from 0 to 1000, relative to the image."
)

EXTRACTION_PROMPT = """

Also return `extraction`: the requested fields pulled from this page, matching
the provided schema. Copy values verbatim from the page; use null for anything
the page does not state, and never invent a value."""

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "kind": {"type": "string", "enum": KINDS},
                    "bbox": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["text", "kind", "bbox"],
                "additionalProperties": False,
            },
        },
        "markdown": {"type": "string"},
    },
    "required": ["blocks", "markdown"],
    "additionalProperties": False,
}


class VisionParser(PdfParser):
    """Base class for one-image-per-page multimodal parsers."""

    kind = "remote"
    tags = ("vision", "remote", "markdown", "bboxes")
    bbox_order = "xyxy"  # or "yxyx"
    # $ per 1M tokens, by model id prefix.
    prices: dict[str, tuple[float, float]] = {}

    options = (
        Option("model", "str", "", help="Model id. Leave blank for the adapter default."),
        Option("max_edge_px", "int", 1600, help="Longest image edge sent to the model."),
        Option("max_output_tokens", "int", 16000),
        Option("include_markdown", "bool", True),
        Option(
            "instructions",
            "text",
            "",
            help="Extra instructions for the model. Applied to every page.",
        ),
        Option(
            "instructions_mode",
            "choice",
            "append",
            choices=["append", "replace"],
            help="'append' adds to the transcription prompt; 'replace' uses your text alone.",
        ),
        Option(
            "extraction_schema",
            "text",
            "",
            help=(
                "JSON Schema for structured extraction. When set, the model also returns an "
                "`extraction` object matching it — the same schema across models is what "
                "makes them comparable."
            ),
        ),
    )

    # -- subclass hook ---------------------------------------------------

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        raise NotImplementedError

    # -- prompt and schema -----------------------------------------------

    @staticmethod
    def parse_schema(raw: Any, label: str = "extraction_schema") -> dict[str, Any] | None:
        """Accept a JSON Schema as either a string or an already-parsed object."""
        if not raw:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            schema = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{label} is not valid JSON: {exc}") from None
        if not isinstance(schema, dict):
            raise RuntimeError(f"{label} must be a JSON object, got {type(schema).__name__}")
        return schema

    def build_prompt(self, opts: dict[str, Any]) -> str:
        base = PROMPT.format(
            kinds=", ".join(f'"{k}"' for k in KINDS),
            bbox_help=BBOX_HELP_YXYX if self.bbox_order == "yxyx" else BBOX_HELP_XYXY,
        )
        if self.parse_schema(opts.get("extraction_schema")) is not None:
            base += EXTRACTION_PROMPT

        instructions = str(opts.get("instructions") or "").strip()
        if not instructions:
            return base
        if (opts.get("instructions_mode") or "append") == "replace":
            return instructions
        return f"{base}\n\n## Additional instructions\n\n{instructions}"

    def build_schema(self, opts: dict[str, Any]) -> dict[str, Any]:
        """The response schema, widened with the caller's extraction schema."""
        extraction = self.parse_schema(opts.get("extraction_schema"))
        if extraction is None:
            return JSON_SCHEMA
        schema = json.loads(json.dumps(JSON_SCHEMA))  # don't mutate the shared one
        schema["properties"]["extraction"] = extraction
        schema["required"] = [*schema["required"], "extraction"]
        return schema

    # -- driver ----------------------------------------------------------

    def parse(self, pdf_path: Path, pages: list[int] | None, options: dict[str, Any]) -> ParsedDocument:
        opts = self.resolved_options(options)
        geometry = page_geometry(pdf_path)
        wanted = select_pages(len(geometry), pages)
        prompt = self.build_prompt(opts)

        out_pages: list[PageResult] = []
        timings: dict[int, float] = {}
        warnings: list[str] = []
        markdown_parts: list[str] = []
        extractions: dict[int, Any] = {}
        total = Usage(requests=0, input_tokens=0, output_tokens=0, cost_usd=0.0)

        for page_no in wanted:
            started = time.perf_counter()
            geo = geometry[page_no]
            scale = float(opts["max_edge_px"]) / max(geo.width, geo.height)
            image = render_page_image(pdf_path, page_no, scale=scale)
            buf = io.BytesIO()
            image.save(buf, format="PNG")

            payload, usage = self.call_model(buf.getvalue(), prompt, opts)
            total.requests += usage.requests or 1
            total.input_tokens = (total.input_tokens or 0) + (usage.input_tokens or 0)
            total.output_tokens = (total.output_tokens or 0) + (usage.output_tokens or 0)
            if usage.cost_usd:
                total.cost_usd = (total.cost_usd or 0.0) + usage.cost_usd
            total.model = usage.model or total.model

            result = PageResult(
                page_number=page_no, width=geo.width, height=geo.height, rotation=geo.rotation
            )
            for i, raw in enumerate(payload.get("blocks") or []):
                text = str(raw.get("text") or "")
                box = self._to_points(raw.get("bbox"), geo.width, geo.height)
                if box is None and raw.get("bbox"):
                    warnings.append(f"page {page_no}: unusable bbox {raw.get('bbox')!r}")
                result.blocks.append(
                    Block(
                        id=f"p{page_no}-v{i}",
                        page=page_no,
                        layer="region",
                        kind=str(raw.get("kind") or "text"),
                        text=text,
                        bbox=box,
                        order=i,
                    )
                )
            result.text = "\n".join(b.text for b in result.blocks if b.text)
            page_md = str(payload.get("markdown") or "")
            if page_md:
                markdown_parts.append(page_md)
            result.meta["markdown"] = page_md
            if payload.get("extraction") is not None:
                extractions[page_no] = payload["extraction"]
                result.meta["extraction"] = payload["extraction"]

            timings[page_no] = time.perf_counter() - started
            out_pages.append(result)

        return ParsedDocument(
            pages=out_pages,
            markdown="\n\n---\n\n".join(markdown_parts) if markdown_parts else None,
            # One request per page, so the document-level extraction is the
            # per-page ones keyed by page rather than a single object.
            extraction={"pages": extractions} if extractions else None,
            usage=total,
            warnings=warnings,
            per_page_s=timings,
        )

    # -- helpers ---------------------------------------------------------

    def _to_points(self, raw: Any, width: float, height: float) -> BBox | None:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            return None
        try:
            a, b, c, d = (float(v) for v in raw)
        except (TypeError, ValueError):
            return None
        if self.bbox_order == "yxyx":
            y0, x0, y1, x1 = a, b, c, d
        else:
            x0, y0, x1, y1 = a, b, c, d
        # Accept 0-1 as well as 0-1000, since models drift between them.
        span = max(abs(v) for v in (x0, y0, x1, y1))
        divisor = 1.0 if span <= 1.5 else 1000.0
        return BBox(
            x0=x0 / divisor * width,
            y0=y0 / divisor * height,
            x1=x1 / divisor * width,
            y1=y1 / divisor * height,
        ).normalized()

    def estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        """Price a call from the table, or report nothing rather than a guess.

        The longest matching prefix wins, so `gpt-4.1-mini` is priced as a mini
        whatever order the table happens to be written in.
        """
        matches = [p for p in self.prices if model.startswith(p)]
        if not matches:
            return None
        in_price, out_price = self.prices[max(matches, key=len)]
        return input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price

    @staticmethod
    def loads(text: str) -> dict[str, Any]:
        """Parse a JSON payload, tolerating a stray ```json fence."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
        return json.loads(text)
