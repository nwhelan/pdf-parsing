"""What comes back, when it isn't what the docs promise.

A remote call fails in more ways than "it raised". A content filter returns a
success with no choices; a token limit returns a success with no content; a
proxy returns a 200 whose body is an error page. Each of those used to surface
as an IndexError, an empty page, or `'list' object has no attribute 'get'` —
none of which say what happened.

Silence is the worst of them: a parser that "succeeded" with nothing scores like
a bad parser rather than a misconfigured call, and a comparison quietly draws
the wrong conclusion.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

import pdfplay.parsers  # noqa: F401  (registers the adapters)
from pdfplay import registry
from pdfplay.parsers.mistral_parser import MistralOCRParser

VALID = '{"blocks": [], "markdown": "# statement"}'


def message(content: Any) -> Any:
    return type("Message", (), {"content": content})()


def choice(content: Any, finish: str = "stop") -> Any:
    return type("Choice", (), {"message": message(content), "finish_reason": finish})()


def response(choices: list, usage: bool = True) -> Any:
    return type(
        "Response",
        (),
        {
            "choices": choices,
            "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 20})() if usage else None,
            "model_dump": lambda self: {"id": "resp"},
        },
    )()


@pytest.fixture
def call(monkeypatch):
    """Run one call_model against a canned response."""
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")

    def run(canned: Any, **options: Any):
        parser = cls()

        class Client:
            chat = property(lambda self: self)
            completions = property(lambda self: self)

            def create(self, **kwargs: Any):
                return canned

        monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: Client()))
        opts = parser.resolved_options({"model": "m", "base_url": "http://x/v1", **options})
        payload, usage = parser.call_model(b"\x89PNG", "prompt", opts)
        return parser, payload, usage

    return run


# -- chat responses that look like success ----------------------------------


def test_no_choices_names_the_content_filter(call):
    """Azure answers 200 with an empty choices list when the prompt is filtered."""
    with pytest.raises(RuntimeError, match="no choices.*content filter"):
        call(response([]))


def test_no_content_because_it_ran_out_of_room_says_so(call):
    """This used to become `{}` — a page reported as successfully empty."""
    with pytest.raises(RuntimeError, match="cut off before any content"):
        call(response([choice(None, finish="length")]))


def test_no_content_because_it_was_filtered_says_so(call):
    with pytest.raises(RuntimeError, match="withheld by a content filter"):
        call(response([choice(None, finish="content_filter")]))


def test_no_content_for_an_unknown_reason_still_reports_the_reason(call):
    with pytest.raises(RuntimeError, match="finish_reason='tool_calls'"):
        call(response([choice("", finish="tool_calls")]))


def test_a_truncated_but_parseable_reply_is_flagged_not_silently_accepted(call):
    """The JSON parsed, but the page is incomplete — and incomplete scores badly."""
    parser, payload, _ = call(response([choice(VALID, finish="length")]))
    assert payload["markdown"] == "# statement", "what did arrive is still returned"
    assert any("truncated" in w for w in parser.parse_warnings)


def test_a_healthy_reply_warns_about_nothing(call):
    parser, payload, usage = call(response([choice(VALID)]))
    assert payload["markdown"] == "# statement"
    assert parser.parse_warnings == []
    assert (usage.input_tokens, usage.output_tokens) == (10, 20)


def test_a_server_that_omits_usage_is_not_an_error(call):
    """Many OpenAI-compatible servers never send one."""
    _, payload, usage = call(response([choice(VALID)], usage=False))
    assert payload["markdown"] == "# statement"
    assert usage.input_tokens is None


def test_the_truncation_warning_reaches_the_result(borderless, monkeypatch):
    """It has to survive out to the user, not just sit on the adapter."""
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()

    class Client:
        chat = property(lambda self: self)
        completions = property(lambda self: self)

        def create(self, **kwargs: Any):
            return response([choice(VALID, finish="length")])

    monkeypatch.setattr(cls, "build_client", classmethod(lambda c, o, env=None: Client()))
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({"model": "m", "base_url": "http://x/v1"}))
    assert any("truncated" in w for w in parsed.warnings)


# -- OCR bodies that aren't OCR responses -----------------------------------


@pytest.fixture
def ocr(monkeypatch, borderless):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")

    def run(body: Any, **options: Any):
        monkeypatch.setattr(MistralOCRParser, "post_ocr", lambda *a, **k: body)
        parser = registry.get("mistral-ocr-3")()
        return parser.parse(borderless.path, [1], parser.resolved_options(options))

    return run


@pytest.mark.parametrize(
    "body,described",
    [
        (["not", "a", "dict"], "answered with list"),
        ("a bare string", "answered with str"),
        (None, "answered with NoneType"),
    ],
)
def test_a_body_that_is_not_an_object_quotes_what_came_back(ocr, body, described):
    """A proxy or a WAF answers 200 with something that is not an OCR response."""
    with pytest.raises(RuntimeError, match=described):
        ocr(body)


def test_an_error_body_returned_with_a_200_lists_what_it_did_contain(ocr):
    with pytest.raises(RuntimeError, match="no 'pages'"):
        ocr({"detail": "Not Found"})


def test_pages_of_the_wrong_type_is_named(ocr):
    with pytest.raises(RuntimeError, match="'pages' came back as str"):
        ocr({"pages": "nope"})


def test_a_single_bad_page_entry_is_skipped_rather_than_fatal(ocr):
    """One malformed entry shouldn't cost you the other nineteen pages."""
    parsed = ocr(
        {
            "pages": [
                "junk",
                {"index": 0, "markdown": "# real", "dimensions": {}, "images": []},
            ]
        }
    )
    assert len(parsed.pages) == 1
    assert parsed.pages[0].text == "# real"
    assert any("not an object" in w for w in parsed.warnings)


def test_a_page_without_markdown_is_tolerated(ocr):
    parsed = ocr({"pages": [{"index": 0}]})
    assert len(parsed.pages) == 1
    assert parsed.pages[0].text == ""


def test_a_healthy_ocr_body_warns_about_nothing(ocr):
    parsed = ocr({"pages": [{"index": 0, "markdown": "# ok", "dimensions": {}, "images": []}]})
    assert [p.text for p in parsed.pages] == ["# ok"]
    assert parsed.warnings == []


# -- requests that could only come back empty -------------------------------


def test_a_page_selection_that_matches_nothing_stops_before_the_call(borderless, monkeypatch):
    """It used to POST `pages: []` — paying for a request that cannot return anything."""
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    calls: list[int] = []
    monkeypatch.setattr(
        MistralOCRParser, "post_ocr", lambda *a, **k: (calls.append(1), {"pages": []})[1]
    )

    parser = registry.get("mistral-ocr-3")()
    with pytest.raises(RuntimeError, match="no such page.*document has 2 pages"):
        parser.parse(borderless.path, [5], parser.resolved_options({}))
    assert calls == [], "and no request was made"


def test_the_vision_path_does_not_report_a_successful_run_of_nothing(borderless, monkeypatch):
    pytest.importorskip("openai")
    cls = registry.get("openai-compatible")
    parser = cls()
    with pytest.raises(RuntimeError, match="no such page"):
        parser.parse(borderless.path, [99], parser.resolved_options({"model": "m", "base_url": "http://x/v1"}))


def test_a_partly_valid_selection_keeps_the_pages_that_exist(borderless):
    parsed = registry.get("pymupdf")().parse(borderless.path, [2, 99], {})
    assert [p.page_number for p in parsed.pages] == [2]


def test_no_selection_still_means_every_page(borderless):
    parsed = registry.get("pymupdf")().parse(borderless.path, None, {})
    assert len(parsed.pages) == 2
