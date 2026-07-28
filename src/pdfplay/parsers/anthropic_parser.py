"""Claude vision adapter (Anthropic Messages API).

Uses structured outputs so the response is guaranteed to match the shared
vision schema, and adaptive thinking (on by default on Opus 5) with a tunable
effort level — layout reading is one of the tasks where effort actually moves
the needle.
"""

from __future__ import annotations

import base64
from typing import Any

from ..models import Usage
from .base import Option
from .vision_base import VisionParser

DEFAULT_MODEL = "claude-opus-5"


class AnthropicVisionParser(VisionParser):
    id = "claude"
    name = "Claude (vision)"
    description = (
        "Sends each page as an image to Claude and asks for a structured transcription with "
        "normalized boxes. Strong on messy layouts and on reading tables as rows; boxes are "
        "approximate compared to a text-layer parser."
    )
    homepage = "https://platform.claude.com/docs/en/build-with-claude/vision"
    requires = ("anthropic",)
    env_vars = ("ANTHROPIC_API_KEY",)
    extra = "anthropic"
    cost_hint = "~$5/$25 per 1M tokens on Opus 5; one request per page"
    prices = {
        "claude-opus-5": (5.0, 25.0),
        "claude-opus-4": (5.0, 25.0),
        "claude-sonnet-5": (3.0, 15.0),
        "claude-sonnet-4": (3.0, 15.0),
        "claude-haiku-4-5": (1.0, 5.0),
        "claude-fable-5": (10.0, 50.0),
    }
    options = VisionParser.options + (
        Option(
            "effort",
            "choice",
            "medium",
            choices=["low", "medium", "high", "xhigh", "max"],
            help="Higher effort spends more thinking tokens on layout reasoning.",
        ),
    )

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        import anthropic

        model = opts.get("model") or DEFAULT_MODEL
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=int(opts["max_output_tokens"]),
            output_config={
                "effort": opts["effort"],
                "format": {"type": "json_schema", "schema": self.build_schema(opts)},
            },
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.standard_b64encode(png).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            raise RuntimeError(f"Claude declined this page (category={category})")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                "hit max_tokens before finishing the page — raise max_output_tokens "
                "or lower effort (thinking shares the same budget)"
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
            requests=1,
        )
        usage.cost_usd = self.estimate_cost(model, usage.input_tokens or 0, usage.output_tokens or 0)
        return self.loads(text), usage
