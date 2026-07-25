"""OpenAI vision adapter."""

from __future__ import annotations

import base64
from typing import Any

from ..models import Usage
from .vision_base import JSON_SCHEMA, VisionParser

DEFAULT_MODEL = "gpt-4.1"


class OpenAIVisionParser(VisionParser):
    id = "openai"
    name = "OpenAI (vision)"
    description = (
        "Page image -> JSON transcription via the Chat Completions API with a strict "
        "json_schema response format."
    )
    homepage = "https://platform.openai.com/docs/guides/vision"
    requires = ("openai",)
    env_vars = ("OPENAI_API_KEY",)
    extra = "openai"
    cost_hint = "billed per request; see OpenAI pricing for the chosen model"

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        from openai import OpenAI

        model = opts.get("model") or DEFAULT_MODEL
        client = OpenAI()
        data_url = "data:image/png;base64," + base64.standard_b64encode(png).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            max_tokens=int(opts["max_output_tokens"]),
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "page_transcription", "schema": JSON_SCHEMA, "strict": True},
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or "{}"
        usage = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            model=model,
            requests=1,
        )
        return self.loads(text), usage
