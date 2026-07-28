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

from .. import litellm_config
from ..models import Usage
from .base import Option
from .vision_base import VisionParser


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
        Option(
            "config_path",
            "str",
            "",
            help=(
                "LiteLLM proxy config.yaml. Blank looks at $PDFPLAY_LITELLM_CONFIG, "
                "$LITELLM_CONFIG_PATH, then ./litellm.config.yaml and ./config.yaml. "
                "A model named in it brings its own endpoint and credentials."
            ),
        ),
        Option("num_retries", "int", 2, help="Retries litellm makes on a failed call."),
    )

    # -- config file -----------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Offer the configured model names as suggestions for `model`.

        The names are read at describe time rather than baked in, so editing
        config.yaml and reloading the viewer is enough to see a new model. The
        option stays free text — a provider/model string that isn't in the
        config still works.
        """
        spec = super().describe()
        names = litellm_config.model_names()
        if names:
            for option in spec["options"]:
                if option["name"] == "model":
                    option["choices"] = names
                    option["help"] = (
                        f"A model_name from {litellm_config.find_config()}, "
                        "or any provider/model string."
                    )
        return spec

    @classmethod
    def configured_params(cls, opts: dict[str, Any]) -> dict[str, Any]:
        """Call parameters from config.yaml for this model, if it names one."""
        model = (opts.get("model") or "").strip()
        if not model:
            return {}
        return litellm_config.resolve_model(model, (opts.get("config_path") or "").strip()) or {}

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
                        "schema": self.build_schema(opts),
                        "strict": not opts.get("extraction_schema"),
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

        # A model_name from config.yaml supplies the real model string, the
        # endpoint and the credentials; anything set explicitly here still wins,
        # so the config is a starting point rather than a cage.
        params = self.configured_params(opts)
        explicit = {
            "model": model,
            "api_base": (opts.get("api_base") or "").strip() or None,
            "api_key": api_key,
        }
        params.update({k: v for k, v in explicit.items() if v is not None and k != "model"})
        if "model" not in params:
            params["model"] = model
        schema_model = params["model"]

        data_url = "data:image/png;base64," + base64.standard_b64encode(png).decode("ascii")
        response = litellm.completion(
            **params,
            max_tokens=int(opts["max_output_tokens"]),
            num_retries=int(opts["num_retries"]),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            # Capability lookup needs the real provider/model, not the friendly name.
            **self.response_format(schema_model, opts),
        )

        text = response.choices[0].message.content or "{}"
        raw_usage = getattr(response, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "prompt_tokens", None),
            output_tokens=getattr(raw_usage, "completion_tokens", None),
            cost_usd=_cost_of(response),
            model=schema_model,
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
