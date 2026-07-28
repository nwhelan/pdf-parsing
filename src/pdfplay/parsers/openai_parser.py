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
"""

from __future__ import annotations

import base64
import os
from typing import Any

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
    options = VisionParser.options + ENDPOINT_OPTIONS

    # -- client ----------------------------------------------------------

    @classmethod
    def build_client(cls, opts: dict[str, Any], env: dict[str, str] | None = None):
        from openai import AzureOpenAI, OpenAI

        env = os.environ if env is None else env
        base_url = (opts.get("base_url") or "").strip()
        named = (opts.get("api_key_env") or "").strip() or "OPENAI_API_KEY"
        key = env.get(named) or ""
        if not key:
            if not base_url:
                raise RuntimeError(f"no API key: set {named}")
            key = PLACEHOLDER_KEY

        version = (opts.get("api_version") or "").strip()
        if version:
            if not base_url:
                raise RuntimeError("Azure OpenAI needs base_url set to the resource endpoint")
            return AzureOpenAI(azure_endpoint=base_url, api_version=version, api_key=key)
        return OpenAI(api_key=key, base_url=base_url or None)

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
        model = (opts.get("model") or "").strip() or self.default_model
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
        return self.loads(text), usage


class OpenAICompatibleParser(OpenAIVisionParser):
    id = "openai-compatible"
    name = "OpenAI-compatible endpoint"
    description = (
        "The Chat Completions adapter with no provider assumptions: point base_url at a "
        "LiteLLM proxy, vLLM, Ollama, OpenRouter, Together, Fireworks or Mistral, name the "
        "model, and it becomes another row in the comparison. Add it once per endpoint you "
        "want to score — options are part of the result cache key."
    )
    homepage = "https://platform.openai.com/docs/api-reference/chat"
    tags = ("vision", "remote", "markdown", "bboxes", "byo-endpoint")
    env_vars = ()  # the key env var is an option, and local servers need none
    default_model = ""  # no sensible default: the endpoint decides
    cost_hint = "depends entirely on the endpoint; free for a local server"
