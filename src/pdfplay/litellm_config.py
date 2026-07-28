"""Read models and endpoints out of a LiteLLM proxy config.

If you already run a LiteLLM proxy, its ``config.yaml`` is where the models,
their deployments and their credentials already live:

.. code-block:: yaml

    model_list:
      - model_name: statement-ocr
        litellm_params:
          model: azure_ai/mistral-document-ai-2512
          api_base: https://my-resource.services.ai.azure.com
          api_key: os.environ/AZURE_AI_API_KEY
          api_version: "2026-01-01"

Retyping that into a sidebar is both tedious and a chance to get it subtly
wrong, so the parser reads the file instead: ``model`` becomes the friendly
``model_name`` and everything under ``litellm_params`` comes with it. Keys given
as ``os.environ/NAME`` are resolved from the environment at call time and never
stored in a result, a preset, or a cache key.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

ENV_PREFIX = "os.environ/"

CONFIG_ENV_VARS = ("PDFPLAY_LITELLM_CONFIG", "LITELLM_CONFIG_PATH")
# Where a proxy config usually sits, relative to the working directory.
CONFIG_CANDIDATES = (
    "litellm.config.yaml",
    "litellm_config.yaml",
    "config.yaml",
    "litellm/config.yaml",
)
# Credentials are resolved per call and never persisted with a result.
SECRET_KEYS = ("api_key", "aws_secret_access_key", "aws_access_key_id", "vertex_credentials")


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
    """``{model_name: litellm_params}`` for every entry in ``model_list``.

    A name may appear several times in a proxy config — that is how load
    balancing across deployments is expressed. Only the first is kept here,
    since this is about naming one endpoint rather than distributing load.
    """
    entries: dict[str, dict[str, Any]] = {}
    for item in load_config(path).get("model_list") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("model_name") or "").strip()
        params = item.get("litellm_params")
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
    """Expand LiteLLM's ``os.environ/NAME`` indirection."""
    if isinstance(value, str) and value.startswith(ENV_PREFIX):
        name = value[len(ENV_PREFIX) :]
        resolved = os.environ.get(name)
        if not resolved:
            raise RuntimeError(f"{value} is referenced by the config but {name} is not set")
        return resolved
    return value


def resolve_model(name: str, explicit: str = "") -> dict[str, Any] | None:
    """The call parameters for a configured model name, or None if unknown.

    Returns litellm's own parameter names, so the result can be splatted
    straight into ``litellm.completion``.
    """
    path = find_config(explicit)
    if path is None:
        return None
    entries = model_entries(path)
    params = entries.get(name)
    if params is None:
        return None
    return {key: resolve_env(value) for key, value in params.items() if value is not None}


def describe(explicit: str = "") -> dict[str, Any]:
    """What the UI needs to say about the config it found."""
    path = find_config(explicit)
    if path is None:
        return {"path": "", "models": []}
    return {"path": str(path), "models": model_names(explicit)}


def redacted(params: dict[str, Any]) -> dict[str, Any]:
    """Call parameters safe to record in a result or a warning."""
    return {k: v for k, v in params.items() if k not in SECRET_KEYS}
