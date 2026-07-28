"""Read models and endpoints out of a LiteLLM-style ``config.yaml``.

The file format is LiteLLM's, because that is what people already have on disk
if they run a proxy — but nothing here imports litellm. It is a YAML file
listing named models and where to reach them:

.. code-block:: yaml

    model_list:
      - model_name: statement-vision
        litellm_params:
          model: azure/gpt-4.1-deployment
          api_base: https://my-resource.openai.azure.com
          api_key: os.environ/AZURE_OPENAI_API_KEY
          api_version: "2026-01-01"

Retyping that into a sidebar is tedious and a chance to get it subtly wrong, so
the OpenAI-compatible parser reads it instead: ``model`` becomes the friendly
``model_name`` and the endpoint comes with it. The parameters map onto the
OpenAI SDK — ``api_base`` is the client's ``base_url``, ``api_version`` selects
the Azure client, and a ``provider/`` prefix on the model is stripped, since the
provider is decided by the URL you are pointing at.

Keys written as ``os.environ/NAME`` are resolved from the environment at call
time and never stored in a result, a preset, or a cache key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_PREFIX = "os.environ/"

CONFIG_ENV_VARS = ("PDFPLAY_MODEL_CONFIG", "PDFPLAY_LITELLM_CONFIG", "LITELLM_CONFIG_PATH")
# Where a proxy config usually sits, relative to the working directory.
CONFIG_CANDIDATES = (
    "litellm.config.yaml",
    "litellm_config.yaml",
    "config.yaml",
    "litellm/config.yaml",
)
SECRET_KEYS = ("api_key", "aws_secret_access_key", "aws_access_key_id", "vertex_credentials")

# Prefixes that name a provider rather than a model. The endpoint decides the
# provider here, so they are stripped from the model string on the way out.
PROVIDER_PREFIXES = (
    "openai/",
    "azure/",
    "azure_ai/",
    "openrouter/",
    "together_ai/",
    "fireworks_ai/",
    "mistral/",
    "ollama/",
    "ollama_chat/",
    "vertex_ai/",
    "hosted_vllm/",
)


def find_config(explicit: str = "") -> Path | None:
    """Locate a config: the given path, then the env vars, then the usual names."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    for name in CONFIG_ENV_VARS:
        value = (os.environ.get(name) or "").strip()
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser()
    for name in CONFIG_CANDIDATES:
        path = Path(name)
        if path.is_file():
            return path
    return None


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def model_entries(path: Path) -> dict[str, dict[str, Any]]:
    """``{model_name: params}`` for every entry in ``model_list``.

    A name may appear several times in a proxy config — that is how load
    balancing across deployments is expressed. Only the first is kept, since
    this is about naming one endpoint rather than distributing load.
    """
    entries: dict[str, dict[str, Any]] = {}
    for item in load_config(path).get("model_list") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("model_name") or "").strip()
        params = item.get("litellm_params") or item.get("params")
        if name and isinstance(params, dict) and name not in entries:
            entries[name] = params
    return entries


def model_names(explicit: str = "") -> list[str]:
    """Configured model names, for offering them as choices. Never raises."""
    path = find_config(explicit)
    if path is None:
        return []
    try:
        return list(model_entries(path))
    except Exception:  # pragma: no cover - a malformed config is not fatal
        return []


def resolve_env(value: Any) -> Any:
    """Expand the ``os.environ/NAME`` indirection."""
    if isinstance(value, str) and value.startswith(ENV_PREFIX):
        name = value[len(ENV_PREFIX) :]
        resolved = os.environ.get(name)
        if not resolved:
            raise RuntimeError(f"{value} is referenced by the config but {name} is not set")
        return resolved
    return value


def strip_provider(model: str) -> str:
    """`azure/gpt-4.1` -> `gpt-4.1`; the endpoint already says who serves it."""
    for prefix in PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def resolve_model(name: str, explicit: str = "") -> dict[str, Any] | None:
    """OpenAI client settings for a configured model name, or None if unknown.

    Returns this project's option names — ``model``, ``base_url``, ``api_key``,
    ``api_version`` — rather than the file's, so the caller has nothing to
    translate.
    """
    path = find_config(explicit)
    if path is None:
        return None
    params = model_entries(path).get(name)
    if params is None:
        return None

    resolved: dict[str, Any] = {}
    if params.get("model"):
        resolved["model"] = strip_provider(str(params["model"]))
    for source, target in (("api_base", "base_url"), ("api_key", "api_key"), ("api_version", "api_version")):
        if params.get(source) is not None:
            resolved[target] = resolve_env(params[source])
    return resolved


def describe(explicit: str = "") -> dict[str, Any]:
    """What the UI needs to say about the config it found."""
    path = find_config(explicit)
    if path is None:
        return {"path": "", "models": []}
    return {"path": str(path), "models": model_names(explicit)}


def redacted(params: dict[str, Any]) -> dict[str, Any]:
    """Settings safe to record in a result or a warning."""
    return {k: v for k, v in params.items() if k not in SECRET_KEYS}
