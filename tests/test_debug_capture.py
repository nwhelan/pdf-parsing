"""What was sent, kept for when the call fails.

An API error is only diagnosable against the request that produced it — a 404 on
a deployment name, a 400 asking for a field you didn't send. By the time the
error reaches the viewer the request is gone, so adapters record it as they go
and the runner keeps it whether the parse returned or raised.

Credentials must not survive that trip: the log is written to disk and read in a
browser.
"""

from __future__ import annotations

import json
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
    assert request["wire"]["url"].endswith("/v1/ocr")
    assert request["wire"]["body"]["model"] == "mistral-ocr-2512"
    assert request["wire"]["body"]["pages"] == [0]
    assert "sk-test" not in str(request), "the Authorization header is masked"
    assert request["wire"]["body"]["document"]["document_url"].startswith("data:application/pdf;base64,")


def test_the_wire_content_is_separated_from_how_it_was_decided(borderless, monkeypatch):
    """`endpoint` is an option of ours, not a field of Mistral's API.

    It used to sit beside the payload in the log, and a reader took it for part
    of the request body. Anything not sent lives under `context`.
    """
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    monkeypatch.setattr(
        MistralOCRParser,
        "post_ocr",
        lambda *a, **k: {"pages": [{"index": 0, "markdown": "x", "dimensions": {}, "images": []}]},
    )

    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({"endpoint": "mistral"}))

    request = next(e for e in parsed.debug if e["event"] == "POST")
    assert set(request["wire"]) == {"url", "headers", "body"}
    assert "endpoint" not in request["wire"]["body"], "not a field the API accepts"
    assert request["context"]["endpoint_option"] == "mistral"


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
    payload = next(e for e in parsed.debug if e["event"] == "POST")["wire"]["body"]
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
    assert request["context"]["model"] == "gpt-4.1"
    assert request["context"]["model_source"] == "option"
    assert request["context"]["image_bytes"] == 4
    assert request["context"]["prompt_chars"] > 0
    # Non-verbose: the message *structure*, not the megabyte of image.
    assert request["wire"]["messages"] == [{"role": "user", "content": ["image_url", "text"]}]
    assert request["wire"]["response_format"]["type"] == "json_schema"


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
    sent = request["wire"]["messages"][0]["content"]
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
    assert request["context"]["model"] == "claude-sonnet-5", "which is what the 404 was about"
    assert request["context"]["base_url"] == "http://x/v1"


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
    assert request["context"]["model_source"] == "config.yaml"
    assert request["context"]["model"] == "gpt-4.1-deployment", "the deployment, not the friendly name"


# -- parameters and endpoints the API will reject ---------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gpt-4.1", "max_tokens"),
        ("gpt-4o", "max_tokens"),
        ("gpt-5.2", "max_completion_tokens"),
        ("gpt-5-mini", "max_completion_tokens"),
        ("o4-mini", "max_completion_tokens"),
        ("o3", "max_completion_tokens"),
    ],
)
def test_the_token_limit_uses_the_name_each_model_accepts(model, expected):
    """GPT-5 and the o-series reject `max_tokens` outright."""
    assert registry.get("openai").token_param(model) == expected


def test_the_token_parameter_can_be_forced():
    cls = registry.get("openai")
    assert cls.token_param("gpt-5.2", "max_tokens") == "max_tokens"
    assert cls.token_param("gpt-4.1", "max_completion_tokens") == "max_completion_tokens"


def test_a_gpt5_call_sends_max_completion_tokens(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    client = FakeChat()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: client))

    parser.call_model(b"\x89PNG", "prompt", parser.resolved_options({"model": "gpt-5.2", "base_url": "http://x/v1"}))

    body = parser.debug_events[0]["wire"]
    assert "max_completion_tokens" in body
    assert "max_tokens" not in body


class PickyAboutTokens(FakeChat):
    """Rejects `max_tokens` the way a GPT-5 deployment does, then accepts."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[dict] = []

    def create(self, **kwargs: Any):
        self.seen.append(kwargs)
        if "max_tokens" in kwargs:
            raise RuntimeError(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            )
        return super().create(**kwargs)


def test_a_rejection_is_retried_with_the_other_spelling(monkeypatch):
    """An Azure deployment can be named anything, so the model id is not enough."""
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    client = PickyAboutTokens()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: client))

    # A deployment name that hides the fact it is a GPT-5 model.
    parser.call_model(
        b"\x89PNG", "prompt", parser.resolved_options({"model": "prod-vision-01", "base_url": "http://x/v1"})
    )

    assert len(client.seen) == 2, "one rejection, one retry"
    assert "max_tokens" in client.seen[0]
    assert "max_completion_tokens" in client.seen[1]
    assert any("retry" in e["event"] for e in parser.debug_events), "and the retry is in the log"


def test_an_unrelated_error_is_not_retried(monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: FakeChat(fail=True)))

    with pytest.raises(RuntimeError, match="DeploymentNotFound"):
        parser.call_model(
            b"\x89PNG", "prompt", parser.resolved_options({"model": "gpt-4.1", "base_url": "http://x/v1"})
        )
    assert len([e for e in parser.debug_events if "retry" in e["event"]]) == 0


def test_an_azure_openai_root_without_a_version_is_called_out():
    """The plain client against a resource root 404s; say so before it does."""
    cls = registry.get("openai-compatible")
    hints = cls.endpoint_hints(
        {"base_url": "https://my-res.openai.azure.com", "api_version": "", "model": "gpt-5.2"}
    )
    assert any("/openai/v1" in h for h in hints)


def test_no_hint_when_the_azure_url_is_already_right():
    cls = registry.get("openai-compatible")
    hints = cls.endpoint_hints(
        {"base_url": "https://my-res.openai.azure.com/openai/v1", "api_version": "", "model": "gpt-5.2"}
    )
    assert not any("/openai/v1" in h for h in hints)


def test_azure_reminds_you_the_model_field_is_a_deployment_name():
    cls = registry.get("openai-compatible")
    hints = cls.endpoint_hints(
        {"base_url": "https://my-res.openai.azure.com", "api_version": "2026-01-01", "model": "gpt-5.2"}
    )
    assert any("deployment" in h for h in hints)


def test_a_plain_endpoint_gets_no_azure_hints():
    cls = registry.get("openai-compatible")
    assert cls.endpoint_hints({"base_url": "", "api_version": "", "model": "gpt-4.1"}) == []
    assert cls.endpoint_hints(
        {"base_url": "http://localhost:4000/v1", "api_version": "", "model": "llava"}
    ) == []


# -- Mistral endpoints ------------------------------------------------------


@pytest.mark.parametrize(
    "base,expected_path",
    [
        # Serverless models-as-a-service and an AI Services resource serve OCR
        # under different paths; the hostname says which.
        ("https://my-deploy.models.ai.azure.com", "/v1/ocr"),
        ("https://my-res.services.ai.azure.com", "/providers/mistral/azure/ocr"),
    ],
)
def test_the_foundry_path_follows_the_hostname(base, expected_path):
    cls = registry.get("mistral-ocr-3")
    url = cls.resolve_endpoint(cls.resolved_options({"endpoint": "azure", "base_url": base}), env={})
    assert url == base + expected_path


def test_a_bare_host_typed_into_base_url_is_completed_too():
    """Posting a document to a hostname is never what was meant."""
    cls = registry.get("mistral-ocr-3")
    url = cls.resolve_endpoint(
        cls.resolved_options({"endpoint": "azure", "base_url": "https://my-res.services.ai.azure.com"}),
        env={},
    )
    assert url.endswith("/providers/mistral/azure/ocr")


def test_a_url_that_already_has_a_path_is_left_alone():
    """It may point at a gateway with its own routing."""
    cls = registry.get("mistral-ocr-3")
    gateway = "https://gw.internal/mistral/proxy"
    assert cls.resolve_endpoint(
        cls.resolved_options({"endpoint": "azure", "base_url": gateway}), env={}
    ) == gateway


def test_pointing_mistral_ocr_at_azure_openai_is_called_out():
    """Azure OpenAI does not serve OCR models at all — that was a real 404."""
    cls = registry.get("mistral-ocr-3")
    opts = cls.resolved_options({"endpoint": "azure"})
    hints = cls.endpoint_hints("https://my-res.openai.azure.com", opts, "mistral-ocr-4-0")

    assert any("does not serve Mistral OCR" in h for h in hints)
    assert any("no path" in h for h in hints)
    assert any("deployment name" in h for h in hints), "mistral-ocr-* is not a Foundry deployment"


def test_a_correct_foundry_configuration_draws_no_complaints():
    cls = registry.get("mistral-ocr-3")
    opts = cls.resolved_options({"endpoint": "azure", "model": "mistral-document-ai-2512"})
    url = "https://my-res.services.ai.azure.com/providers/mistral/azure/ocr"
    assert cls.endpoint_hints(url, opts, "mistral-document-ai-2512") == []


def test_hints_reach_the_debug_log_and_the_warnings(borderless, monkeypatch):
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
        parser.resolved_options({"endpoint": "azure", "base_url": "https://my-res.openai.azure.com"}),
    )

    request = next(e for e in parsed.debug if e["event"] == "POST")
    assert any("does not serve Mistral OCR" in h for h in request["hints"])
    assert any("does not serve Mistral OCR" in w for w in parsed.warnings)


@pytest.mark.parametrize("key", ["token_param", "max_completion_tokens", "input_tokens", "monkey"])
def test_parameter_names_that_merely_contain_a_secret_word_survive(key):
    """A redactor that hides `token_param` is less useful without being safer."""
    assert redact({key: "max_completion_tokens"}) == {key: "max_completion_tokens"}


# -- credentials that don't live under a telling key ------------------------


@pytest.mark.parametrize(
    "value",
    [
        "https://host/v1/ocr?api-version=1&api-key=sk-LEAKED",
        "https://host/v1/ocr?subscription-key=sk-LEAKED",
        "https://func.azurewebsites.net/api/x?code=sk-LEAKED",
        "https://host/v1?access_token=sk-LEAKED",
        "https://user:sk-LEAKED@host/v1/ocr",
        "sent Authorization: Bearer sk-LEAKED",
    ],
)
def test_a_credential_inside_a_value_is_masked_too(value):
    """Gateways and Azure take keys in the URL; key-name matching alone misses them."""
    assert "sk-LEAKED" not in str(redact({"url": value}))


def test_scrubbing_leaves_the_parts_you_need_to_read():
    url = "https://my-res.services.ai.azure.com/providers/mistral/azure/ocr?api-version=2026-01-01"
    assert redact({"url": url})["url"] == url, "api-version is not a secret"


# -- the log must never be able to break the result it belongs to -----------


@pytest.mark.parametrize("value", [b"\xff\xfe", object(), {1, 2}, Exception("boom")])
def test_anything_recorded_survives_json_serialization(value):
    """A log entry that can't be serialized would fail the *save*, losing a paid-for run."""
    json.dumps(redact({"body": value}))


def test_an_unserializable_debug_entry_does_not_lose_the_result(workspace: Workspace, borderless):
    from pdfplay.models import ParsedDocument

    class Sloppy(PdfParser):
        id = "sloppy"
        name = "Sloppy"
        description = "Records raw bytes."

        def parse(self, pdf_path, pages, options):
            self.record_request("POST", wire={"body": b"\xff\xfe raw bytes"})
            return ParsedDocument(debug=self.debug_events)

    registry.register(Sloppy)
    try:
        meta = workspace.add_document(borderless.path)
        result, key, _ = run_parser(workspace, meta.doc_id, "sloppy")
        assert result.status == "ok"
        assert workspace.load_result(meta.doc_id, key) is not None, "it was cached, not lost"
    finally:
        registry._REGISTRY.pop("sloppy", None)


def test_a_failure_to_cache_is_reported_but_does_not_destroy_the_run(
    workspace: Workspace, borderless, monkeypatch
):
    """The API call was already paid for; a full disk must not throw the answer away."""

    def no_room(*args, **kwargs):
        raise OSError("No space left on device")

    monkeypatch.setattr(type(workspace), "save_result", no_room)
    meta = workspace.add_document(borderless.path)

    result, _, _ = run_parser(workspace, meta.doc_id, "pymupdf")

    assert result.status == "ok", "the run still succeeded"
    assert result.pages, "and its output is intact"
    assert any("could not be cached" in w for w in result.warnings)


# -- a model that doesn't answer in JSON ------------------------------------


@pytest.mark.parametrize(
    "body,expected",
    [
        ("I'm sorry, I can't help with that.", "prose rather than JSON"),
        ('{"blocks": [{"text": "a"', "cut off"),
        ("", "empty response"),
    ],
)
def test_a_non_json_reply_is_explained_rather_than_thrown_raw(body, expected):
    """`Expecting value: line 1 column 1` describes none of these."""
    from pdfplay.parsers.vision_base import VisionParser

    with pytest.raises(RuntimeError, match=expected):
        VisionParser.loads(body)


def test_the_explanation_quotes_what_came_back():
    from pdfplay.parsers.vision_base import VisionParser

    with pytest.raises(RuntimeError, match="cannot transcribe"):
        VisionParser.loads("I cannot transcribe documents containing personal data.")


def test_a_fenced_reply_still_parses():
    from pdfplay.parsers.vision_base import VisionParser

    assert VisionParser.loads('```json\n{"blocks": [], "markdown": "x"}\n```')["markdown"] == "x"


# -- misconfigurations that produce an unhelpful 401 ------------------------


def test_a_placeholder_key_going_somewhere_hosted_is_called_out():
    cls = registry.get("openai-compatible")
    settings = cls.settings(
        cls.resolved_options({"base_url": "https://api.openai.com/v1", "model": "gpt-4.1"}), env={}
    )
    assert any("placeholder" in h for h in cls.endpoint_hints(settings))


@pytest.mark.parametrize(
    "base", ["http://localhost:11434/v1", "http://127.0.0.1:4000", "http://ollama.internal/v1"]
)
def test_a_local_server_without_a_key_is_left_alone(base):
    cls = registry.get("openai-compatible")
    settings = cls.settings(cls.resolved_options({"base_url": base, "model": "llava"}), env={})
    assert cls.endpoint_hints(settings) == []


def test_cost_survives_a_server_that_omits_usage():
    """Plenty of OpenAI-compatible servers never send a usage block."""
    assert registry.get("openai")().estimate_cost("gpt-4.1", None, None) == 0.0
