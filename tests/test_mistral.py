"""Mistral OCR adapter: endpoint/auth resolution and response mapping.

The network call is stubbed. What is under test is everything around it —
which URL and headers a given set of options produces, and how the documented
response shape lands in the normalized model.
"""

from __future__ import annotations

from typing import Any

import pytest

import pdfplay.parsers  # noqa: F401  (registers the adapters)
from pdfplay import registry
from pdfplay.metrics import bank_statement
from pdfplay.models import ParseResult
from pdfplay.parsers.mistral_parser import MISTRAL_URL, MistralOCRParser


def opts(**overrides: Any) -> dict[str, Any]:
    return MistralOCRParser.resolved_options(overrides)


# -- endpoint resolution ----------------------------------------------------


def test_default_endpoint_is_the_mistral_api():
    assert MistralOCRParser.resolve_endpoint(opts(), env={}) == MISTRAL_URL


def test_explicit_base_url_wins_over_everything():
    url = MistralOCRParser.resolve_endpoint(
        opts(base_url="https://gateway.internal/ocr", endpoint="azure"),
        env={"AZURE_MISTRAL_ENDPOINT": "https://ignored.example"},
    )
    assert url == "https://gateway.internal/ocr"


def test_azure_resource_root_gets_the_ocr_path_appended():
    url = MistralOCRParser.resolve_endpoint(
        opts(endpoint="azure"),
        env={"AZURE_MISTRAL_ENDPOINT": "https://my-resource.services.ai.azure.com/"},
    )
    assert url == "https://my-resource.services.ai.azure.com/providers/mistral/azure/ocr"


def test_a_full_azure_url_is_left_alone():
    full = "https://my-resource.services.ai.azure.com/providers/mistral/azure/ocr"
    assert MistralOCRParser.resolve_endpoint(opts(endpoint="azure"), env={"AZURE_MISTRAL_ENDPOINT": full}) == full


def test_api_version_is_appended_once():
    url = MistralOCRParser.resolve_endpoint(
        opts(base_url="https://x/ocr", api_version="2026-01-01"), env={}
    )
    assert url == "https://x/ocr?api-version=2026-01-01"
    already = MistralOCRParser.resolve_endpoint(
        opts(base_url="https://x/ocr?api-version=9", api_version="2026-01-01"), env={}
    )
    assert already == "https://x/ocr?api-version=9"


def test_azure_without_a_url_fails_loudly():
    with pytest.raises(RuntimeError, match="needs a URL"):
        MistralOCRParser.resolve_endpoint(opts(endpoint="azure"), env={})


# -- auth -------------------------------------------------------------------


def test_mistral_gets_a_bearer_token_and_azure_gets_api_key():
    env = {"MISTRAL_API_KEY": "sk-test"}
    assert MistralOCRParser.resolve_auth(opts(), env=env) == {"Authorization": "Bearer sk-test"}
    assert MistralOCRParser.resolve_auth(opts(endpoint="azure"), env=env) == {"api-key": "sk-test"}


def test_auth_header_can_be_forced_for_a_gateway():
    env = {"MISTRAL_API_KEY": "sk-test"}
    assert MistralOCRParser.resolve_auth(opts(endpoint="azure", auth_header="bearer"), env=env) == {
        "Authorization": "Bearer sk-test"
    }


def test_a_named_env_var_is_used_exclusively():
    env = {"MISTRAL_API_KEY": "wrong", "TENANT_B_KEY": "right"}
    assert MistralOCRParser.resolve_auth(opts(api_key_env="TENANT_B_KEY"), env=env) == {
        "Authorization": "Bearer right"
    }
    with pytest.raises(RuntimeError, match="no API key"):
        MistralOCRParser.resolve_auth(opts(api_key_env="MISSING"), env=env)


# -- response mapping -------------------------------------------------------


LEDGER_MARKDOWN = "\n".join(
    [
        "# MERIDIAN COMMUNITY BANK",
        "",
        "|Date|Description|Withdrawals|Deposits|Balance|",
        "|---|---|---|---|---|",
        "|03/01/2025|DIRECT DEP ACME CORP PAYROLL||1,708.14|5,110.20|",
        "|03/02/2025|CHECK 1042|31.61||5,078.59|",
    ]
)


class FakeMistral(MistralOCRParser):
    id = "fake-mistral"
    name = "Fake Mistral"
    model_default = "mistral-ocr-test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post_ocr(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload, "timeout": timeout})
        return {
            "model": "mistral-ocr-test-completion",
            "usage_info": {"pages_processed": 1},
            "pages": [
                {
                    "index": 0,
                    "markdown": LEDGER_MARKDOWN,
                    "dimensions": {"dpi": 200, "width": 1700, "height": 2200},
                    "images": [
                        {
                            "id": "img-0",
                            "top_left_x": 850,
                            "top_left_y": 0,
                            "bottom_right_x": 1700,
                            "bottom_right_y": 1100,
                        }
                    ],
                }
            ],
        }


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    return FakeMistral()


def test_the_request_carries_the_pdf_and_the_model(fake, borderless):
    fake.parse(borderless.path, [1], fake.resolved_options({"model": "mistral-ocr-9", "transport": "http"}))
    call = fake.calls[0]

    assert call["url"] == MISTRAL_URL
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    assert call["payload"]["model"] == "mistral-ocr-9"
    assert call["payload"]["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert call["payload"]["pages"] == [0], "the API counts pages from zero"


def test_blank_model_falls_back_to_the_parsers_default(fake, borderless):
    fake.parse(borderless.path, [1], fake.resolved_options({"transport": "http"}))
    assert fake.calls[0]["payload"]["model"] == "mistral-ocr-test"


def test_markdown_becomes_page_text_tables_and_a_ledger(fake, borderless):
    parsed = fake.parse(borderless.path, [1], fake.resolved_options({"transport": "http"}))
    page = parsed.pages[0]

    assert page.page_number == 1, "response index 0 is page 1"
    assert page.width > 0 and page.height > 0, "page size comes from the PDF, not the response"
    assert parsed.markdown and "MERIDIAN" in parsed.markdown

    assert len(page.tables) == 1
    assert (page.tables[0].n_rows, page.tables[0].n_cols) == (3, 5)

    report = bank_statement.analyze(ParseResult(parser_id="fake", status="ok", pages=parsed.pages))
    assert report.n_transactions == 2
    assert report.reconciliation_rate == 1.0


def test_image_boxes_are_rescaled_from_pixels_into_points(fake, borderless):
    parsed = fake.parse(borderless.path, [1], fake.resolved_options({"transport": "http"}))
    page = parsed.pages[0]
    figures = [b for b in page.blocks if b.kind == "figure"]

    assert len(figures) == 1
    box = figures[0].bbox
    # The image covers the right half and top half of a 1700x2200px page.
    assert box.x0 == pytest.approx(page.width / 2, abs=1)
    assert box.x1 == pytest.approx(page.width, abs=1)
    assert box.y0 == pytest.approx(0, abs=1)
    assert box.y1 == pytest.approx(page.height / 2, abs=1)


def test_cost_is_reported_per_page(fake, borderless):
    parsed = fake.parse(borderless.path, [1], fake.resolved_options({"price_per_1k_pages": 2.0, "transport": "http"}))
    assert parsed.usage.cost_usd == pytest.approx(0.002)
    assert parsed.usage.requests == 1


def test_a_page_the_pdf_does_not_have_is_warned_about_not_crashed(fake, borderless, monkeypatch):
    monkeypatch.setattr(
        FakeMistral, "post_ocr", lambda *a, **k: {"pages": [{"index": 999, "markdown": "x"}]}
    )
    parsed = fake.parse(borderless.path, None, fake.resolved_options({"transport": "http"}))
    assert parsed.pages == []
    assert any("does not have" in w for w in parsed.warnings)


# -- litellm transport ------------------------------------------------------


class FakeOCRResponse:
    """Stands in for litellm's OCRResponse: a pydantic model plus hidden params."""

    def __init__(self, body: dict[str, Any], cost: float | None) -> None:
        self._body = body
        self._hidden_params = {"response_cost": cost} if cost is not None else {}

    def model_dump(self) -> dict[str, Any]:
        return self._body


@pytest.fixture
def fake_litellm(monkeypatch):
    """Patch litellm.ocr and record what it was called with."""
    litellm = pytest.importorskip("litellm")
    calls: list[dict[str, Any]] = []

    def ocr(**kwargs):
        calls.append(kwargs)
        return FakeOCRResponse(
            {
                "model": "mistral-ocr-2512",
                "usage_info": {"pages_processed": 1},
                "pages": [{"index": 0, "markdown": LEDGER_MARKDOWN, "dimensions": {}, "images": []}],
            },
            cost=0.0042,
        )

    monkeypatch.setattr(litellm, "ocr", ocr)
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    return calls


def test_auto_transport_prefers_litellm_when_it_is_installed(fake_litellm, borderless):
    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(borderless.path, [1], parser.resolved_options({}))

    assert len(fake_litellm) == 1, "the request went through litellm, not httpx"
    assert parsed.pages[0].tables[0].n_rows == 3, "the response maps the same either way"


def test_the_endpoint_option_selects_the_litellm_provider(fake_litellm, borderless):
    parser = registry.get("mistral-ocr-3")()
    parser.parse(
        borderless.path,
        [1],
        parser.resolved_options(
            {
                "transport": "litellm",
                "endpoint": "azure",
                "base_url": "https://my-resource.services.ai.azure.com",
                "model": "mistral-document-ai-2512",
            }
        ),
    )
    call = fake_litellm[0]

    assert call["model"] == "azure_ai/mistral-document-ai-2512"
    assert call["api_base"] == "https://my-resource.services.ai.azure.com", (
        "litellm completes the OCR path itself, so it gets the bare resource root"
    )
    assert call["api_key"] == "sk-test"
    assert call["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert call["pages"] == [0]


def test_a_provider_prefixed_model_is_passed_through(fake_litellm, borderless):
    parser = registry.get("mistral-ocr-3")()
    parser.parse(
        borderless.path,
        [1],
        parser.resolved_options({"transport": "litellm", "model": "vertex_ai/mistral-ocr-2505"}),
    )
    assert fake_litellm[0]["model"] == "vertex_ai/mistral-ocr-2505"


def test_litellms_measured_cost_beats_the_estimate(fake_litellm, borderless):
    parser = registry.get("mistral-ocr-3")()
    parsed = parser.parse(
        borderless.path,
        [1],
        parser.resolved_options({"transport": "litellm", "price_per_1k_pages": 999.0}),
    )
    assert parsed.usage.cost_usd == pytest.approx(0.0042)


def test_http_transport_is_still_reachable_by_name(monkeypatch, borderless):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-test")
    fake = FakeMistral()
    fake.parse(borderless.path, [1], fake.resolved_options({"transport": "http"}))
    assert fake.calls[0]["url"] == MISTRAL_URL


# -- registration -----------------------------------------------------------


@pytest.mark.parametrize(
    "parser_id,model", [("mistral-ocr-3", "mistral-ocr-2512"), ("mistral-ocr-4", "mistral-ocr-4-0")]
)
def test_both_versions_are_registered_with_their_own_model(parser_id, model):
    cls = registry.get(parser_id)
    assert cls.model_default == model
    assert cls.kind == "remote"
    assert {o.name for o in cls.options} >= {
        "model",
        "endpoint",
        "base_url",
        "api_key_env",
        "auth_header",
        "transport",
    }


@pytest.mark.parametrize("parser_id", ["mistral-ocr-3", "mistral-ocr-4"])
def test_the_default_models_are_ids_a_provider_actually_serves(parser_id):
    """Guard the model strings against litellm's price table.

    Mistral names OCR releases by date rather than by generation, so the ids
    here are easy to get wrong — and a wrong one is a 404 at the first call.
    """
    litellm = pytest.importorskip("litellm")
    model = registry.get(parser_id).model_default
    assert f"mistral/{model}" in litellm.model_cost


def test_availability_names_the_env_vars_it_looked_for(monkeypatch):
    for name in ("MISTRAL_API_KEY", "AZURE_MISTRAL_API_KEY", "MISTRAL_OCR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    avail = registry.get("mistral-ocr-3").check_availability()
    assert not avail.available
    assert "MISTRAL_API_KEY" in avail.reason

    monkeypatch.setenv("AZURE_MISTRAL_API_KEY", "sk-azure")
    assert registry.get("mistral-ocr-3").check_availability().available
