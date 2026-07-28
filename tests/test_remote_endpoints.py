"""Pointing the remote adapters at somewhere other than their vendor.

Everything reachable over the Chat Completions protocol goes through the OpenAI
SDK — `OpenAI` for anything speaking it, `AzureOpenAI` when an api-version says
Azure. No network call is made here; what's under test is the client each set of
options produces.
"""

from __future__ import annotations

from typing import Any

import pytest

import pdfplay.parsers  # noqa: F401  (registers the adapters)
from pdfplay import registry
from pdfplay.parsers.openai_parser import PLACEHOLDER_KEY, OpenAICompatibleParser, OpenAIVisionParser


def opts(cls, **overrides: Any) -> dict[str, Any]:
    return cls.resolved_options(overrides)


# -- OpenAI-compatible client -----------------------------------------------


def test_the_stock_openai_parser_still_talks_to_openai():
    pytest.importorskip("openai")
    client = OpenAIVisionParser.build_client(opts(OpenAIVisionParser), env={"OPENAI_API_KEY": "sk-a"})
    assert "api.openai.com" in str(client.base_url)
    assert client.api_key == "sk-a"


def test_base_url_points_the_same_adapter_at_a_litellm_proxy():
    pytest.importorskip("openai")
    client = OpenAICompatibleParser.build_client(
        opts(OpenAICompatibleParser, base_url="http://localhost:4000/v1"),
        env={"OPENAI_API_KEY": "sk-a"},
    )
    assert str(client.base_url).rstrip("/") == "http://localhost:4000/v1"


def test_a_local_server_needs_no_key_but_a_hosted_one_does():
    pytest.importorskip("openai")
    local = OpenAICompatibleParser.build_client(
        opts(OpenAICompatibleParser, base_url="http://localhost:11434/v1"), env={}
    )
    assert local.api_key == PLACEHOLDER_KEY

    with pytest.raises(RuntimeError, match="no API key"):
        OpenAICompatibleParser.build_client(opts(OpenAICompatibleParser), env={})


def test_a_named_env_var_selects_the_key():
    pytest.importorskip("openai")
    client = OpenAICompatibleParser.build_client(
        opts(OpenAICompatibleParser, base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_KEY"),
        env={"OPENAI_API_KEY": "wrong", "OPENROUTER_KEY": "right"},
    )
    assert client.api_key == "right"


def test_api_version_switches_to_the_azure_client():
    pytest.importorskip("openai")
    from openai import AzureOpenAI

    client = OpenAICompatibleParser.build_client(
        opts(
            OpenAICompatibleParser,
            base_url="https://my-resource.openai.azure.com",
            api_version="2026-01-01",
        ),
        env={"OPENAI_API_KEY": "sk-a"},
    )
    assert isinstance(client, AzureOpenAI)

    with pytest.raises(RuntimeError, match="needs base_url"):
        OpenAICompatibleParser.build_client(
            opts(OpenAICompatibleParser, api_version="2026-01-01"), env={"OPENAI_API_KEY": "sk-a"}
        )


def test_response_format_can_be_lowered_for_servers_that_reject_schemas():
    strict = OpenAIVisionParser.response_format(opts(OpenAIVisionParser))
    assert strict["response_format"]["type"] == "json_schema"
    assert strict["response_format"]["json_schema"]["strict"] is True

    loose = OpenAIVisionParser.response_format(opts(OpenAIVisionParser, response_format="json_object"))
    assert loose == {"response_format": {"type": "json_object"}}

    assert OpenAIVisionParser.response_format(opts(OpenAIVisionParser, response_format="text")) == {}


def test_the_compatible_parser_is_available_without_an_openai_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert registry.get("openai-compatible").check_availability().available
    assert not registry.get("openai").check_availability().available


# -- cost -------------------------------------------------------------------


def test_cost_is_estimated_from_the_price_table():
    """Without a router computing it, cost comes from a per-model rate here."""
    parser = OpenAIVisionParser()
    # gpt-4.1 at $2/$8 per 1M tokens.
    assert parser.estimate_cost("gpt-4.1", 1_000_000, 1_000_000) == pytest.approx(10.0)
    assert parser.estimate_cost("gpt-4.1-mini", 1_000_000, 0) == pytest.approx(0.4)


def test_the_longest_matching_prefix_wins_regardless_of_table_order():
    """`gpt-4.1-mini` must be priced as a mini, not at the `gpt-4.1` rate."""
    parser = OpenAIVisionParser()
    assert parser.estimate_cost("gpt-4.1-mini", 1_000_000, 0) < parser.estimate_cost(
        "gpt-4.1", 1_000_000, 0
    )

    class Reordered(OpenAIVisionParser):
        prices = {"gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6)}

    assert Reordered().estimate_cost("gpt-4.1-mini", 1_000_000, 0) == pytest.approx(0.4)


def test_an_unlisted_model_reports_no_cost_rather_than_a_wrong_one():
    assert OpenAIVisionParser().estimate_cost("some-local-llava", 1_000_000, 1_000_000) is None
