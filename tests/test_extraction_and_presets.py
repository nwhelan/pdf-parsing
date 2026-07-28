"""Prompts, extraction schemas, and saved configurations.

What makes two LLMs comparable on a document is that they were asked the same
question in the same shape. These are the three pieces of that: instructions you
control, a schema the answer must match, and a way to save the combination so
the same comparison can be run again tomorrow.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import pdfplay.parsers  # noqa: F401  (registers the adapters)
from pdfplay import registry
from pdfplay.parsers.mistral_parser import MistralOCRParser
from pdfplay.workspace import Workspace

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "account_number": {"type": "string"},
        "closing_balance": {"type": "number"},
    },
    "required": ["account_number", "closing_balance"],
}


def vision_opts(**overrides: Any) -> dict[str, Any]:
    return registry.get("openai").resolved_options(overrides)


# -- instructions -----------------------------------------------------------


def test_by_default_the_prompt_is_the_transcription_one():
    parser = registry.get("openai")()
    prompt = parser.build_prompt(vision_opts())
    assert "document parsing engine" in prompt
    assert "Additional instructions" not in prompt


def test_instructions_are_appended_to_the_standard_prompt():
    parser = registry.get("openai")()
    prompt = parser.build_prompt(vision_opts(instructions="Treat every dash as a minus sign."))
    assert "document parsing engine" in prompt
    assert prompt.endswith("Treat every dash as a minus sign.")
    assert "## Additional instructions" in prompt


def test_replace_mode_hands_the_model_only_your_text():
    parser = registry.get("openai")()
    prompt = parser.build_prompt(
        vision_opts(instructions="List the transactions.", instructions_mode="replace")
    )
    assert prompt == "List the transactions."


def test_blank_instructions_do_not_alter_the_prompt_in_either_mode():
    """Whitespace in the box must not silently replace the prompt with nothing."""
    parser = registry.get("openai")()
    default = parser.build_prompt(vision_opts())
    for mode in ("append", "replace"):
        assert parser.build_prompt(vision_opts(instructions="  ", instructions_mode=mode)) == default


# -- extraction schema ------------------------------------------------------


def test_the_schema_widens_the_response_rather_than_replacing_it():
    """Transcription and extraction come back from one call, so both are scored."""
    parser = registry.get("openai")()
    schema = parser.build_schema(vision_opts(extraction_schema=json.dumps(INVOICE_SCHEMA)))

    assert set(schema["properties"]) == {"blocks", "markdown", "extraction"}
    assert schema["properties"]["extraction"] == INVOICE_SCHEMA
    assert "extraction" in schema["required"]


def test_the_shared_default_schema_is_not_mutated():
    parser = registry.get("openai")()
    parser.build_schema(vision_opts(extraction_schema=json.dumps(INVOICE_SCHEMA)))
    assert set(parser.build_schema(vision_opts())["properties"]) == {"blocks", "markdown"}


def test_asking_for_extraction_tells_the_model_so():
    parser = registry.get("openai")()
    prompt = parser.build_prompt(vision_opts(extraction_schema=json.dumps(INVOICE_SCHEMA)))
    assert "`extraction`" in prompt
    assert "never invent a value" in prompt


def test_a_schema_that_is_not_json_fails_with_a_readable_message():
    parser = registry.get("openai")()
    with pytest.raises(RuntimeError, match="extraction_schema is not valid JSON"):
        parser.build_schema(vision_opts(extraction_schema="{not json}"))


def test_strict_mode_is_dropped_for_a_caller_supplied_schema():
    """Strict json_schema demands every property be required; a pasted schema won't be."""
    cls = registry.get("openai")
    default = cls.response_format(vision_opts(), None)
    assert default["response_format"]["json_schema"]["strict"] is True

    widened = cls.response_format(vision_opts(), {"type": "object"})
    assert widened["response_format"]["json_schema"]["strict"] is False


def test_a_page_extraction_reaches_the_parsed_document(borderless, monkeypatch):
    """The model's `extraction` object is carried out to the result."""
    parser = registry.get("openai")()

    def fake_call(png, prompt, opts):
        assert "`extraction`" in prompt
        from pdfplay.models import Usage

        return {
            "blocks": [],
            "markdown": "# statement",
            "extraction": {"account_number": "12345", "closing_balance": 5078.59},
        }, Usage(requests=1)

    monkeypatch.setattr(type(parser), "call_model", staticmethod(fake_call))
    parsed = parser.parse(
        borderless.path, [1], parser.resolved_options({"extraction_schema": json.dumps(INVOICE_SCHEMA)})
    )
    assert parsed.extraction == {"pages": {1: {"account_number": "12345", "closing_balance": 5078.59}}}


# -- Mistral's own extraction format ----------------------------------------


def test_a_bare_schema_is_wrapped_in_the_json_schema_envelope():
    params = MistralOCRParser.annotation_params(
        MistralOCRParser.resolved_options({"document_annotation_schema": json.dumps(INVOICE_SCHEMA)})
    )
    envelope = params["document_annotation_format"]
    assert envelope["type"] == "json_schema"
    assert envelope["json_schema"]["schema"] == INVOICE_SCHEMA
    assert envelope["json_schema"]["name"] == "document_annotation"


def test_an_already_wrapped_schema_is_left_alone():
    wrapped = {"type": "json_schema", "json_schema": {"name": "mine", "schema": INVOICE_SCHEMA}}
    params = MistralOCRParser.annotation_params(
        MistralOCRParser.resolved_options({"document_annotation_schema": json.dumps(wrapped)})
    )
    assert params["document_annotation_format"] == wrapped


def test_no_schema_means_no_annotation_parameters():
    assert MistralOCRParser.annotation_params(MistralOCRParser.resolved_options({})) == {}


def test_the_prompt_and_bbox_schema_travel_with_it():
    params = MistralOCRParser.annotation_params(
        MistralOCRParser.resolved_options(
            {
                "document_annotation_prompt": "Only the ledger.",
                "bbox_annotation_schema": json.dumps({"type": "object"}),
            }
        )
    )
    assert params["document_annotation_prompt"] == "Only the ledger."
    assert params["bbox_annotation_format"]["json_schema"]["name"] == "bbox_annotation"


def test_annotation_parameters_are_sent_and_the_answer_comes_back(monkeypatch, borderless):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    calls: list[dict[str, Any]] = []

    def post_ocr(self, url, headers, payload, timeout):
        calls.append(payload)
        return {
            "model": "mistral-ocr-2512",
            "pages": [{"index": 0, "markdown": "# statement", "dimensions": {}, "images": []}],
            # Mistral returns the annotation as a JSON string.
            "document_annotation": '{"account_number": "12345", "closing_balance": 5078.59}',
        }

    monkeypatch.setattr(MistralOCRParser, "post_ocr", post_ocr)

    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(
        borderless.path,
        [1],
        parser.resolved_options(
            {
                "document_annotation_schema": json.dumps(INVOICE_SCHEMA),
                "document_annotation_prompt": "Only the ledger.",
            }
        ),
    )

    sent = calls[0]
    assert sent["document_annotation_format"]["json_schema"]["schema"] == INVOICE_SCHEMA
    assert sent["document_annotation_prompt"] == "Only the ledger."
    assert parsed.extraction == {"account_number": "12345", "closing_balance": 5078.59}


def test_an_unparseable_annotation_is_warned_about_not_dropped(monkeypatch, borderless):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")

    monkeypatch.setattr(
        MistralOCRParser,
        "post_ocr",
        lambda *a, **k: {
            "model": "m",
            "pages": [{"index": 0, "markdown": "x", "dimensions": {}, "images": []}],
            "document_annotation": "sorry, I could not comply",
        },
    )
    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({}))

    assert parsed.extraction == "sorry, I could not comply"
    assert any("not valid JSON" in w for w in parsed.warnings)


# -- presets ----------------------------------------------------------------


def test_a_preset_round_trips_through_the_workspace(workspace: Workspace):
    saved = workspace.save_preset(
        "Foundry OCR",
        "mistral-ocr-3",
        {"endpoint": "azure", "model": "mistral-document-ai-2512"},
    )
    assert saved.preset_id == "mistral-ocr-3__foundry-ocr"

    loaded = workspace.get_preset(saved.preset_id)
    assert loaded.options["model"] == "mistral-document-ai-2512"
    assert workspace.get_preset("Foundry OCR").preset_id == saved.preset_id, "findable by name too"


def test_saving_the_same_name_updates_rather_than_duplicates(workspace: Workspace):
    workspace.save_preset("Foundry", "mistral-ocr-3", {"model": "old"})
    workspace.save_preset("Foundry", "mistral-ocr-3", {"model": "new"})
    presets = workspace.list_presets()
    assert len(presets) == 1
    assert presets[0].options["model"] == "new"


def test_presets_are_listed_per_parser_and_deletable(workspace: Workspace):
    workspace.save_preset("Foundry", "mistral-ocr-3", {})
    workspace.save_preset("Local llava", "openai-compatible", {"model": "llava"})

    assert len(workspace.list_presets()) == 2
    assert [p.name for p in workspace.list_presets("openai-compatible")] == ["Local llava"]

    workspace.delete_preset("openai-compatible__local-llava")
    assert [p.parser_id for p in workspace.list_presets()] == ["mistral-ocr-3"]


def test_a_preset_survives_characters_the_ansi_code_page_cannot_hold(workspace: Workspace):
    workspace.save_preset("Extraction — v2", "openai", {"instructions": "Prix en €, dates en JJ/MM"})
    assert workspace.get_preset("openai__extraction-v2").options["instructions"].endswith("JJ/MM")


def test_an_unnamed_preset_is_refused(workspace: Workspace):
    with pytest.raises(ValueError, match="needs a name"):
        workspace.save_preset("   ", "openai", {})


def test_the_api_exposes_presets(client):
    created = client.post(
        "/api/presets",
        json={"name": "GPT-4.1", "parser_id": "openai", "options": {"model": "gpt-4.1"}},
    ).json()
    assert created["preset_id"] == "openai__gpt-4-1"

    listed = client.get("/api/presets", params={"parser_id": "openai"}).json()
    assert [p["name"] for p in listed] == ["GPT-4.1"]
    assert client.get("/api/presets", params={"parser_id": "openai-compatible"}).json() == []

    assert client.delete(f"/api/presets/{created['preset_id']}").json() == {"ok": True}
    assert client.get("/api/presets").json() == []


def test_the_api_rejects_a_nameless_preset(client):
    assert client.post("/api/presets", json={"parser_id": "openai", "options": {}}).status_code == 400
