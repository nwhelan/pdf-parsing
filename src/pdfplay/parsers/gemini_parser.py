"""Gemini vision adapter.

Gemini is asked for boxes in its native ``[ymin, xmin, ymax, xmax]`` 0-1000
convention, which it is trained on and localizes better with than xyxy.
"""

from __future__ import annotations

from typing import Any

from ..models import Usage
from .vision_base import JSON_SCHEMA, VisionParser

DEFAULT_MODEL = "gemini-2.5-pro"


class GeminiVisionParser(VisionParser):
    id = "gemini"
    name = "Gemini (vision)"
    description = (
        "Page image -> JSON transcription via google-genai, using Gemini's native "
        "[ymin, xmin, ymax, xmax] 0-1000 box convention."
    )
    homepage = "https://ai.google.dev/gemini-api/docs/document-processing"
    requires = ("google.genai",)
    env_vars = ("GEMINI_API_KEY",)
    extra = "gemini"
    cost_hint = "billed per request; see Google AI pricing for the chosen model"
    bbox_order = "yxyx"

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        from google import genai
        from google.genai import types

        model = opts.get("model") or DEFAULT_MODEL
        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=png, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=JSON_SCHEMA,
                max_output_tokens=int(opts["max_output_tokens"]),
            ),
        )
        meta = getattr(response, "usage_metadata", None)
        usage = Usage(
            input_tokens=getattr(meta, "prompt_token_count", None),
            output_tokens=getattr(meta, "candidates_token_count", None),
            model=model,
            requests=1,
        )
        return self.loads(response.text or "{}"), usage
