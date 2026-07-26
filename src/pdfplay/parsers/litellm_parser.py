"""LiteLLM vision adapter: one row per model string, any provider.

The other vision adapters each wrap one SDK, which is the right shape when you
want that provider's own features. This one trades that for reach: give it
``anthropic/claude-sonnet-4-5``, ``gemini/gemini-2.5-pro``, ``azure/my-deployment``,
``bedrock/...``, ``ollama/llama3.2-vision`` or ``openrouter/...`` and it runs, so
adding a model to the comparison is a string rather than a new adapter.

Two things come along for free and are worth the dependency on their own:
credentials resolve from each provider's conventional env vars, and cost comes
back measured from litellm's price table instead of estimated from a hard-coded
rate.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from ..models import Usage
from .base import Option
from .vision_base import JSON_SCHEMA, VisionParser


class LiteLLMVisionParser(VisionParser):
    id = "litellm"
    name = "LiteLLM (any provider)"
    description = (
        "Any vision model litellm can reach, named as provider/model — anthropic/, gemini/, "
        "azure/, bedrock/, vertex_ai/, mistral/, openrouter/, ollama/ and so on. Credentials "
        "come from each provider's usual env vars, and the reported cost is litellm's, not an "
        "estimate. Point api_base at a LiteLLM proxy to route through a gateway instead."
    )
    homepage = "https://docs.litellm.ai/docs/providers"
    tags = ("vision", "remote", "markdown", "bboxes", "byo-endpoint")
    requires = ("litellm",)
    extra = "litellm"
    cost_hint = "whatever the routed provider charges; litellm reports it per request"
    options = VisionParser.options + (
        Option(
            "api_base",
            "str",
            "",
            help="Override the provider's URL, e.g. a LiteLLM proxy at http://localhost:4000.",
        ),
        Option(
            "api_key_env",
            "str",
            "",
            help="Env var holding the key. Blank lets litellm use the provider's usual one.",
        ),
        Option(
            "response_format",
            "choice",
            "json_schema",
            choices=["json_schema", "json_object", "text"],
            help="Dropped automatically if the routed model doesn't support it.",
        ),
        Option("num_retries", "int", 2, help="Retries litellm makes on a failed call."),
    )

    # Keys are per-provider and the provider isn't known until the model option
    # is set, so availability only checks that litellm is importable.

    def response_format(self, model: str, opts: dict[str, Any]) -> dict[str, Any]:
        """Ask for structured output only where the routed model supports it."""
        import litellm

        choice = opts.get("response_format") or "json_schema"
        if choice == "text":
            return {}
        try:
            supported = litellm.get_supported_openai_params(model=model) or []
        except Exception:  # pragma: no cover - provider lookup is best effort
            supported = []
        if "response_format" not in supported:
            return {}
        if choice == "json_schema" and litellm.supports_response_schema(model=model):
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "page_transcription",
                        "schema": JSON_SCHEMA,
                        "strict": True,
                    },
                }
            }
        return {"response_format": {"type": "json_object"}}

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        import litellm

        model = (opts.get("model") or "").strip()
        if not model:
            raise RuntimeError(
                "set the model option to a litellm model string, e.g. "
                "'anthropic/claude-sonnet-4-5' or 'gemini/gemini-2.5-pro'"
            )

        named = (opts.get("api_key_env") or "").strip()
        api_key = os.environ.get(named) if named else None
        if named and not api_key:
            raise RuntimeError(f"no API key: {named} is not set")

        data_url = "data:image/png;base64," + base64.standard_b64encode(png).decode("ascii")
        response = litellm.completion(
            model=model,
            max_tokens=int(opts["max_output_tokens"]),
            num_retries=int(opts["num_retries"]),
            api_base=(opts.get("api_base") or "").strip() or None,
            api_key=api_key,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            **self.response_format(model, opts),
        )

        text = response.choices[0].message.content or "{}"
        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", None),
            output_tokens=getattr(raw_usage, "completion_tokens", None),
            cost_usd=_cost_of(response),
            model=model,
            requests=1,
        )
        return self.loads(text), usage


def _cost_of(response: Any) -> float | None:
    """litellm attaches the computed cost; fall back to computing it."""
    cost = getattr(response, "_hidden_params", {}).get("response_cost")
    if cost is not None:
        return float(cost)
    try:
        import litellm

        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # pragma: no cover - unknown model, no price
        return None
