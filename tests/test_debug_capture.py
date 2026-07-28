"""What was sent, kept for when the call fails.

An API error is only diagnosable against the request that produced it — a 404 on
a deployment name, a 400 asking for a field you didn't send. By the time the
error reaches the viewer the request is gone, so adapters record it as they go
and the runner keeps it whether the parse returned or raised.

Credentials must not survive that trip: the log is written to disk and read in a
browser.
"""

from __future__ import annotations

from typing import Any

import pytest

import pdfplay.parsers  # noqa: F401  (registers the adapters)
from pdfplay import registry
from pdfplay.parsers.base import MAX_DEBUG_VALUE, PdfParser, redact
from pdfplay.parsers.mistral_parser import MistralOCRParser
from pdfplay.runner import run_parser
from pdfplay.workspace import Workspace


# -- redaction --------------------------------------------------------------


@pytest.mark.parametrize(
    "key", ["Authorization", "api-key", "api_key", "OPENAI_API_KEY", "access_token", "client_secret"]
)
def test_anything_that_looks_like_a_credential_is_masked(key):
    masked = redact({key: "sk-super-secret-value"}, "")
    assert "sk-super-secret" not in str(masked)
    assert "redacted" in str(masked[key])


def test_masking_reaches_into_nested_structures():
    masked = redact({"headers": {"Authorization": "Bearer sk-abc"}, "list": [{"api_key": "sk-def"}]})
    assert "sk-abc" not in str(masked)
    assert "sk-def" not in str(masked)


def test_a_base64_document_is_truncated_to_its_shape():
    payload = {"document": {"document_url": "data:application/pdf;base64," + "A" * 50_000}}
    masked = redact(payload)
    url = masked["document"]["document_url"]
    assert url.startswith("data:application/pdf;base64,"), "the shape is still readable"
    assert len(url) < MAX_DEBUG_VALUE + 100
    assert "+49" in url or "+50" in url, "and it says how much was dropped"


def test_ordinary_values_pass_through_untouched():
    payload = {"model": "gpt-4.1", "max_tokens": 16000, "pages": [0, 1], "strict": True}
    assert redact(payload) == payload


# -- the runner keeps the log across a failure ------------------------------


class Recorder(PdfParser):
    """An adapter that records a request and then fails, like a real one would."""

    id = "recorder"
    name = "Recorder"
    description = "Records a request, then fails."

    def parse(self, pdf_path, pages, options):
        self.record_request("POST", url="https://example.invalid/v1/ocr", headers={"api-key": "sk-x"})
        raise RuntimeError("Error code: 404 - DeploymentNotFound")


@pytest.fixture
def recorder():
    """Register the stub for one test, then take it back out of the registry."""
    registry.register(Recorder)
    yield Recorder
    registry._REGISTRY.pop(Recorder.id, None)


def test_the_request_survives_the_failure_that_followed_it(recorder, workspace: Workspace, borderless):
    """The whole point: the error and the request that caused it arrive together."""
    meta = workspace.add_document(borderless.path)

    result, _, _ = run_parser(workspace, meta.doc_id, "recorder")

    assert result.status == "error"
    assert "DeploymentNotFound" in result.error
    assert len(result.debug) == 1
    assert result.debug[0]["url"] == "https://example.invalid/v1/ocr"
    assert "sk-x" not in str(result.debug), "the key does not reach the log"


def test_the_log_round_trips_through_the_workspace(recorder, workspace: Workspace, borderless):
    """It is read back from disk in the viewer, so it has to persist."""
    meta = workspace.add_document(borderless.path)
    _, key, _ = run_parser(workspace, meta.doc_id, "recorder")

    reloaded = workspace.load_result(meta.doc_id, key)
    assert reloaded is not None
    assert reloaded.debug[0]["event"] == "POST"


def test_each_run_starts_with_a_clean_log(recorder, workspace: Workspace, borderless):
    """A fresh adapter per run, so one failure's request can't be attributed to the next."""
    meta = workspace.add_document(borderless.path)
    first, _, _ = run_parser(workspace, meta.doc_id, "recorder", force=True)
    second, _, _ = run_parser(workspace, meta.doc_id, "recorder", force=True)
    assert len(first.debug) == len(second.debug) == 1


def test_a_parser_that_records_nothing_reports_an_empty_log(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    result, _, _ = run_parser(workspace, meta.doc_id, "pymupdf")
    assert result.status == "ok"
    assert result.debug == []


# -- what the adapters record -----------------------------------------------


def test_mistral_records_the_url_and_payload_it_posted(borderless, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setattr(
        MistralOCRParser,
        "post_ocr",
        lambda *a, **k: {"pages": [{"index": 0, "markdown": "x", "dimensions": {}, "images": []}]},
    )

    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({"endpoint": "mistral"}))

    request = next(e for e in parsed.debug if e["event"] == "POST")
    assert request["url"].endswith("/v1/ocr")
    assert request["payload"]["model"] == "mistral-ocr-2512"
    assert request["payload"]["pages"] == [0]
    assert "sk-test" not in str(request), "the Authorization header is masked"
    assert request["payload"]["document"]["document_url"].startswith("data:application/pdf;base64,")


def test_the_annotation_parameters_appear_in_the_log(borderless, monkeypatch):
    """A 400 for a missing annotation prompt is only readable against what was sent."""
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setattr(
        MistralOCRParser,
        "post_ocr",
        lambda *a, **k: {"pages": [{"index": 0, "markdown": "x", "dimensions": {}, "images": []}]},
    )

    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(
        borderless.path,
        [1],
        parser.resolved_options({"document_annotation_schema": '{"type": "object"}'}),
    )
    payload = next(e for e in parsed.debug if e["event"] == "POST")["payload"]
    assert payload["document_annotation_format"]["type"] == "json_schema"
    assert "document_annotation_prompt" not in payload, "not sent, so visibly absent"


def test_a_failing_mistral_call_still_leaves_the_request(workspace: Workspace, borderless, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")

    def explode(*args, **kwargs):
        raise RuntimeError("returned 400: Please provide a document_annotation_prompt")

    monkeypatch.setattr(MistralOCRParser, "post_ocr", explode)
    meta = workspace.add_document(borderless.path)

    result, _, _ = run_parser(workspace, meta.doc_id, "mistral-ocr-3")

    assert result.status == "error"
    assert "document_annotation_prompt" in result.error
    assert result.debug[0]["event"] == "POST", "the request that got the 400 is right there"


class FakeChat:
    """Enough of the OpenAI client for call_model to run."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs: Any):
        if self.fail:
            raise RuntimeError("Error code: 404 - DeploymentNotFound")
        message = type("M", (), {"content": '{"blocks": [], "markdown": "hi"}'})()
        return type(
            "R",
            (),
            {
                "choices": [type("C", (), {"message": message})()],
                "usage": type("U", (), {"prompt_tokens": 5, "completion_tokens": 6})(),
                "model_dump": lambda self: {"id": "resp-1"},
            },
        )()


def test_openai_records_the_client_endpoint_and_request_shape(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat()))

    parser.call_model(
        b"\x89PNG", "prompt", parser.resolved_options({"model": "gpt-4.1", "base_url": "http://x/v1"})
    )

    request = next(e for e in parser.debug_events if e["event"] == "chat.completions.create")
    assert request["model"] == "gpt-4.1"
    assert request["model_source"] == "option"
    assert request["image_bytes"] == 4
    assert request["prompt_chars"] > 0
    # Non-verbose: the message *structure*, not the megabyte of image.
    assert request["request"]["messages"] == [{"role": "user", "content": ["image_url", "text"]}]
    assert request["request"]["response_format"]["type"] == "json_schema"


def test_debug_mode_keeps_the_prompt_and_the_response_body(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat()))

    parser.call_model(
        b"\x89PNG",
        "transcribe this",
        parser.resolved_options({"model": "gpt-4.1", "base_url": "http://x/v1", "debug": True}),
    )

    request = next(e for e in parser.debug_events if e["event"] == "chat.completions.create")
    sent = request["request"]["messages"][0]["content"]
    assert sent[1]["text"] == "transcribe this", "the actual prompt, verbatim"
    assert sent[0]["image_url"]["url"].startswith("data:image/png;base64,")

    response = next(e for e in parser.debug_events if e["event"] == "chat.completions")
    assert response["body"] == {"id": "resp-1"}


def test_without_debug_mode_the_response_body_is_not_kept(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat()))

    parser.call_model(b"\x89PNG", "prompt", parser.resolved_options({"model": "gpt-4.1", "base_url": "http://x/v1"}))
    response = next(e for e in parser.debug_events if e["event"] == "chat.completions")
    assert "body" not in response
    assert response["keys"] == ["id"], "the shape is still reported"


def test_a_404_from_the_model_leaves_the_request_behind(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat(fail=True)))

    with pytest.raises(RuntimeError, match="DeploymentNotFound"):
        parser.call_model(
            b"\x89PNG", "prompt", parser.resolved_options({"model": "claude-sonnet-5", "base_url": "http://x/v1"})
        )

    request = parser.debug_events[0]
    assert request["model"] == "claude-sonnet-5", "which is what the 404 was about"
    assert request["base_url"] == "http://x/v1"


def test_a_configured_model_says_where_it_came_from(monkeypatch, tmp_path):
    """`model_source` answers 'did it read my config.yaml or my option?'"""
    pytest.importorskip("openai")
    pytest.importorskip("yaml")

    config = tmp_path / "config.yaml"
    config.write_text(
        "model_list:\n"
        "  - model_name: statement-vision\n"
        "    litellm_params:\n"
        "      model: azure/gpt-4.1-deployment\n"
        "      api_base: https://res.openai.azure.com\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PDFPLAY_MODEL_CONFIG", str(config))

    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat()))

    parser.call_model(b"\x89PNG", "prompt", parser.resolved_options({"model": "statement-vision"}))

    request = parser.debug_events[0]
    assert request["model_source"] == "config.yaml"
    assert request["model"] == "gpt-4.1-deployment", "the deployment, not the friendly name"
