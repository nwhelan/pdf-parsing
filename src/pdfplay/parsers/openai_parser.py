"""OpenAI vision adapter, and the same adapter pointed anywhere.

The Chat Completions shape is the closest thing this space has to a lingua
franca: vLLM, Ollama, OpenRouter, Together, Fireworks, a LiteLLM proxy and
Mistral all speak it, and Azure OpenAI speaks a near-identical dialect. So the
endpoint is an option rather than a constant, and ``openai-compatible`` is the
same adapter registered a second time with no provider assumptions — one
comparison row per server you want to try.

Two things vary between servers and are therefore options too:

``response_format``
    Strict ``json_schema`` is an OpenAI feature that self-hosted servers often
    only partly implement. Fall back to ``json_object``, or to ``text`` plus
    the parser's own fence-tolerant JSON loader, when a server rejects it.
``api_version``
    Set it and the Azure OpenAI client is used instead, with ``base_url`` read
    as the resource endpoint and the model read as a deployment name.

If you keep a LiteLLM-style ``config.yaml`` of named models and endpoints, the
``model`` option can name an entry in it and the endpoint, key and api-version
come along — see :mod:`pdfplay.model_config`. Nothing here depends on litellm
itself; the file is read as YAML and mapped onto the OpenAI client.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from .. import model_config
from ..models import Usage
from .base import Option
from .vision_base import JSON_SCHEMA, VisionParser

DEFAULT_MODEL = "gpt-4.1"

ENDPOINT_OPTIONS = (
    Option(
        "base_url",
        "str",
        "",
        help="OpenAI-compatible base URL, e.g. http://localhost:4000/v1. Blank uses OpenAI.",
    ),
    Option(
        "api_key_env",
        "str",
        "",
        help="Env var holding the key. Blank uses OPENAI_API_KEY.",
    ),
    Option(
        "api_version",
        "str",
        "",
        help="Set for Azure OpenAI. Switches to the Azure client; base_url is the resource endpoint.",
    ),
    Option(
        "response_format",
        "choice",
        "json_schema",
        choices=["json_schema", "json_object", "text"],
        help="Lower this if a server rejects strict schemas. 'text' relies on the prompt alone.",
    ),
    Option(
        "config_path",
        "str",
        "",
        help=(
            "A config.yaml of named models and endpoints. Blank looks at "
            "$PDFPLAY_MODEL_CONFIG, $LITELLM_CONFIG_PATH, then ./config.yaml. A model named "
            "in it brings its own endpoint and credentials."
        ),
    ),
)

# Servers behind a base_url are often local and unauthenticated, but the SDK
# insists on a key being present.
PLACEHOLDER_KEY = "not-needed"


class OpenAIVisionParser(VisionParser):
    id = "openai"
    name = "OpenAI (vision)"
    description = (
        "Page image -> JSON transcription via the Chat Completions API with a strict "
        "json_schema response format. Set base_url to reach any OpenAI-compatible server."
    )
    homepage = "https://platform.openai.com/docs/guides/vision"
    requires = ("openai",)
    env_vars = ("OPENAI_API_KEY",)
    extra = "openai"
    cost_hint = "billed per request; see OpenAI pricing for the chosen model"
    default_model = DEFAULT_MODEL
    # $ per 1M tokens (input, output), matched by model-id prefix. Only used to
    # report an estimate — an unlisted model reports no cost rather than a wrong one.
    prices = {
        "gpt-5": (1.25, 10.0),
        "gpt-4.1-mini": (0.4, 1.6),
        "gpt-4.1-nano": (0.1, 0.4),
        "gpt-4.1": (2.0, 8.0),
        "gpt-4o-mini": (0.15, 0.6),
        "gpt-4o": (2.5, 10.0),
        "o4-mini": (1.1, 4.4),
    }
    options = VisionParser.options + ENDPOINT_OPTIONS

    # -- config file -----------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """Offer the configured model names as suggestions for `model`.

        Read at describe time rather than baked in, so editing config.yaml and
        reloading the viewer is enough to see a new model. The option stays free
        text — a model id the config doesn't mention still works.
        """
        spec = super().describe()
        names = model_config.model_names()
        if names:
            for option in spec["options"]:
                if option["name"] == "model":
                    option["choices"] = names
                    option["help"] = (
                        f"A model_name from {model_config.find_config()}, "
                        "or any model id this endpoint serves."
                    )
        return spec

    # -- client ----------------------------------------------------------

    @classmethod
    def configured(cls, opts: dict[str, Any]) -> dict[str, Any]:
        """Settings from config.yaml when `model` names an entry there."""
        model = (opts.get("model") or "").strip()
        if not model:
            return {}
        return model_config.resolve_model(model, (opts.get("config_path") or "").strip()) or {}

    @classmethod
    def settings(cls, opts: dict[str, Any], env: dict[str, str] | None = None) -> dict[str, Any]:
        """Endpoint, key, api-version and model, from the config then the options.

        The config is a starting point rather than a cage: anything set
        explicitly on the parser wins over the file.
        """
        env = os.environ if env is None else env
        resolved = cls.configured(opts)

        base_url = (opts.get("base_url") or "").strip() or resolved.get("base_url", "")
        version = (opts.get("api_version") or "").strip() or resolved.get("api_version", "")
        model = resolved.get("model") or (opts.get("model") or "").strip() or cls.default_model

        named = (opts.get("api_key_env") or "").strip()
        key = env.get(named) if named else None
        if named and not key:
            raise RuntimeError(f"no API key: {named} is not set")
        if not key:
            key = resolved.get("api_key") or env.get("OPENAI_API_KEY") or ""
        if not key:
            if not base_url:
                raise RuntimeError("no API key: set OPENAI_API_KEY, or name another var in api_key_env")
            # A server behind a base_url is often local and unauthenticated,
            # but the SDK insists on a key being present.
            key = PLACEHOLDER_KEY

        return {"model": model, "base_url": base_url, "api_key": key, "api_version": str(version)}

    @classmethod
    def build_client(cls, opts: dict[str, Any], env: dict[str, str] | None = None):
        from openai import AzureOpenAI, OpenAI

        settings = cls.settings(opts, env)
        if settings["api_version"]:
            if not settings["base_url"]:
                raise RuntimeError("Azure OpenAI needs base_url set to the resource endpoint")
            return AzureOpenAI(
                azure_endpoint=settings["base_url"],
                api_version=settings["api_version"],
                api_key=settings["api_key"],
            )
        return OpenAI(api_key=settings["api_key"], base_url=settings["base_url"] or None)

    @classmethod
    def response_format(cls, opts: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]:
        choice = opts.get("response_format") or "json_schema"
        if choice == "json_schema":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "page_transcription",
                        "schema": schema or JSON_SCHEMA,
                        # A user-supplied extraction schema won't satisfy strict
                        # mode's requirements (every property required, no extra
                        # properties), so strictness follows the default schema.
                        "strict": schema is None,
                    },
                }
            }
        if choice == "json_object":
            return {"response_format": {"type": "json_object"}}
        return {}

    # -- driver ----------------------------------------------------------

    def call_model(self, png: bytes, prompt: str, opts: dict[str, Any]) -> tuple[dict[str, Any], Usage]:
        settings = self.settings(opts)
        model = settings["model"]
        if not model:
            raise RuntimeError("set the model option to the model this endpoint serves")

        client = self.build_client(opts)
        data_url = "data:image/png;base64," + base64.standard_b64encode(png).decode("ascii")
        response = client.chat.completions.create(
            model=model,
            max_tokens=int(opts["max_output_tokens"]),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            **self.response_format(opts, self.build_schema(opts)),
        )
        text = response.choices[0].message.content or "{}"
        usage = Usage(
            input_tokens=getattr(response.usage, "prompt_tokens", None),
            output_tokens=getattr(response.usage, "completion_tokens", None),
            model=model,
            requests=1,
        )
        usage.cost_usd = self.estimate_cost(model, usage.input_tokens or 0, usage.output_tokens or 0)
        return self.loads(text), usage


class OpenAICompatibleParser(OpenAIVisionParser):
    id = "openai-compatible"
    name = "OpenAI-compatible endpoint"
    description = (
        "The Chat Completions adapter with no provider assumptions: point base_url at Azure "
        "OpenAI, a LiteLLM proxy, vLLM, Ollama, OpenRouter, Together, Fireworks or Mistral, "
        "name the model, and it becomes another row in the comparison. Or name a model from "
        "a config.yaml and the endpoint comes with it."
    )
    homepage = "https://platform.openai.com/docs/api-reference/chat"
    tags = ("vision", "remote", "markdown", "bboxes", "byo-endpoint")
    env_vars = ()  # the key env var is an option, and local servers need none
    default_model = ""  # no sensible default: the endpoint decides
    cost_hint = "depends entirely on the endpoint; free for a local server"
