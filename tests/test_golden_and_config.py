"""Scoring extractions against a golden set, and reading a LiteLLM config.

The golden scorer is the labelled counterpart to the bank-statement one: given
the right answer for a document, it says which fields each parser got. Its
judgement calls are the interesting part — a parser writing `$1,708.14` where
the golden says `1708.14` has not made a mistake, and a scorer that says it has
will send you chasing nothing.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from pdfplay import litellm_config
from pdfplay.metrics import extraction
from pdfplay.workspace import Workspace

GOLDEN = {
    "account_number": "0042-118-9",
    "statement_date": "2025-03-31",
    "closing_balance": 5078.59,
    "holder": {"name": "A. Nwosu"},
    "fees": [{"label": "Monthly fee", "amount": 12.0}],
}


def score(actual, golden=GOLDEN):
    return extraction.score_against_golden(actual, golden)


def status_of(report, path):
    return next(f["status"] for f in report["fields"] if f["path"] == path)


# -- value comparison -------------------------------------------------------


def test_a_perfect_answer_scores_one():
    report = score(GOLDEN)
    # Six leaves: the four scalars plus holder.name and the two fee fields.
    assert (report["n_correct"], report["n_fields"]) == (6, 6)
    assert report["accuracy"] == report["precision"] == report["recall"] == report["f1"] == 1.0


@pytest.mark.parametrize(
    "written",
    ["5078.59", "$5,078.59", " 5,078.59 ", 5078.5901],
    ids=["string", "currency", "padded", "rounding"],
)
def test_the_same_amount_written_differently_is_the_same_amount(written):
    """Parsers disagree about formatting; the scorer is about the value."""
    report = score({**GOLDEN, "closing_balance": written})
    assert status_of(report, "closing_balance") == "correct"


def test_a_negative_in_accounting_parentheses_reads_as_negative():
    assert extraction.as_number("(1,708.14)") == -1708.14
    assert extraction.values_match(-1708.14, "(1,708.14)")


def test_a_wrong_amount_is_wrong_however_close():
    report = score({**GOLDEN, "closing_balance": 5078.69})
    assert status_of(report, "closing_balance") == "wrong"


def test_dates_match_across_formats():
    for written in ("03/31/2025", "2025-03-31", "31/03/25"):
        assert extraction.values_match("2025-03-31", written), written


def test_text_ignores_case_spacing_and_punctuation():
    report = score({**GOLDEN, "holder": {"name": "a  nwosu"}})
    assert status_of(report, "holder.name") == "correct"


def test_a_number_and_a_sentence_never_match():
    """Punctuation-stripping must not turn '5078 USD' into the number 5078."""
    assert not extraction.values_match(5078.59, "5078.59 USD")


def test_a_near_miss_records_how_near():
    report = score({**GOLDEN, "holder": {"name": "A. Nwoso"}})
    field = next(f for f in report["fields"] if f["path"] == "holder.name")
    assert field["status"] == "wrong"
    assert 0.8 < field["similarity"] < 1.0


# -- what counts as answered ------------------------------------------------


def test_a_missing_field_is_distinguished_from_a_wrong_one():
    report = score({k: v for k, v in GOLDEN.items() if k != "account_number"})
    assert status_of(report, "account_number") == "missing"
    assert report["recall"] == pytest.approx(5 / 6, abs=1e-3)
    assert report["precision"] == 1.0, "not answering is not the same as answering wrongly"


def test_a_null_answer_counts_as_missing_not_as_a_match():
    report = score({**GOLDEN, "account_number": None})
    assert status_of(report, "account_number") == "missing"


def test_answering_nothing_scores_zero_recall_rather_than_perfect_precision():
    report = score({})
    assert report["recall"] == 0.0
    assert report["precision"] is None
    assert report["f1"] is None


def test_fields_the_golden_does_not_mention_are_reported_as_extra():
    report = score({**GOLDEN, "invented": "hallucination"})
    assert status_of(report, "invented") == "extra"
    assert report["recall"] == 1.0
    assert report["precision"] < 1.0, "an invented field costs precision"


def test_nested_lists_are_addressed_by_index():
    report = score({**GOLDEN, "fees": [{"label": "Monthly fee", "amount": 99.0}]})
    assert status_of(report, "fees[0].amount") == "wrong"
    assert status_of(report, "fees[0].label") == "correct"


# -- shapes -----------------------------------------------------------------


def test_a_per_page_extraction_is_merged_before_scoring():
    """Vision parsers answer per page; Mistral answers per document."""
    per_page = {
        "pages": {
            1: {"account_number": "0042-118-9", "closing_balance": None, "holder": {"name": "A. Nwosu"}},
            2: {"account_number": None, "closing_balance": 5078.59, "statement_date": "2025-03-31"},
        }
    }
    merged = extraction.normalize(per_page)
    assert merged["closing_balance"] == 5078.59
    assert merged["account_number"] == "0042-118-9"

    report = extraction.score_against_golden(per_page, {k: GOLDEN[k] for k in ("account_number", "closing_balance")})
    assert report["accuracy"] == 1.0


def test_repeated_structures_accumulate_across_pages_rather_than_being_replaced():
    per_page = {"pages": {1: {"fees": [{"label": "A"}]}, 2: {"fees": [{"label": "B"}]}}}
    assert extraction.normalize(per_page)["fees"] == [{"label": "A"}, {"label": "B"}]


def test_a_document_level_extraction_is_left_alone():
    assert extraction.normalize(GOLDEN) == GOLDEN
    assert extraction.normalize({"pages": "not a mapping"}) == {"pages": "not a mapping"}


def test_a_parser_that_extracted_nothing_still_scores():
    report = score(None)
    assert report["n_correct"] == 0
    assert report["recall"] == 0.0
    assert all(f["status"] == "missing" for f in report["fields"])


# -- storage and the API ----------------------------------------------------


def test_the_golden_sits_beside_the_ledger_in_one_file(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    workspace.set_ground_truth(meta.doc_id, {"transactions": [{"amount": 1.0}]})
    workspace.set_golden_extraction(meta.doc_id, GOLDEN)

    truth = workspace.get_ground_truth(meta.doc_id)
    assert truth["transactions"] == [{"amount": 1.0}], "the ledger survives"
    assert workspace.get_golden_extraction(meta.doc_id)["closing_balance"] == 5078.59


def test_no_golden_reads_as_none(workspace: Workspace, borderless):
    meta = workspace.add_document(borderless.path)
    assert workspace.get_golden_extraction(meta.doc_id) is None


def test_the_api_round_trips_a_golden_and_scores_against_it(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]

    assert client.put(f"/api/documents/{doc_id}/golden-extraction", json=GOLDEN).status_code == 200
    assert client.get(f"/api/documents/{doc_id}/golden-extraction").json()["extraction"] == GOLDEN

    client.post(f"/api/documents/{doc_id}/parse/pymupdf", json={})
    scores = client.post(f"/api/documents/{doc_id}/score", json={}).json()

    assert scores["has_golden_extraction"] is True
    report = scores["rows"][0]["extraction_score"]
    assert report["n_fields"] == 6
    # pymupdf has no extraction at all, so every field is missing rather than wrong.
    assert report["n_correct"] == 0
    assert {f["status"] for f in report["fields"]} == {"missing"}


def test_scoring_says_nothing_about_extraction_without_a_golden(client):
    doc_id = client.get("/api/documents").json()[0]["doc_id"]
    client.post(f"/api/documents/{doc_id}/parse/pymupdf", json={})
    scores = client.post(f"/api/documents/{doc_id}/score", json={}).json()
    assert scores["has_golden_extraction"] is False
    assert "extraction_score" not in scores["rows"][0]


# -- LiteLLM config ---------------------------------------------------------


CONFIG = textwrap.dedent(
    """
    model_list:
      - model_name: statement-ocr
        litellm_params:
          model: azure_ai/mistral-document-ai-2512
          api_base: https://my-resource.services.ai.azure.com
          api_key: os.environ/AZURE_AI_API_KEY
          api_version: "2026-01-01"
      - model_name: cheap-vision
        litellm_params:
          model: gemini/gemini-2.5-flash
          api_key: os.environ/GEMINI_API_KEY
      - model_name: statement-ocr
        litellm_params:
          model: azure_ai/a-second-deployment
    """
)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    path = tmp_path / "litellm.config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("PDFPLAY_LITELLM_CONFIG", str(path))
    monkeypatch.setenv("AZURE_AI_API_KEY", "sk-azure")
    return path


def test_model_names_come_from_the_config(config_file):
    assert litellm_config.model_names() == ["statement-ocr", "cheap-vision"]


def test_a_configured_model_brings_its_endpoint_and_key(config_file):
    params = litellm_config.resolve_model("statement-ocr")
    assert params["model"] == "azure_ai/mistral-document-ai-2512"
    assert params["api_base"] == "https://my-resource.services.ai.azure.com"
    assert params["api_version"] == "2026-01-01"
    assert params["api_key"] == "sk-azure", "os.environ/NAME is resolved at call time"


def test_the_first_deployment_of_a_duplicated_name_wins(config_file):
    """A repeated model_name is proxy load balancing; here it names one endpoint."""
    assert litellm_config.resolve_model("statement-ocr")["model"].endswith("mistral-document-ai-2512")


def test_an_unset_referenced_variable_says_which_one(config_file, monkeypatch):
    monkeypatch.delenv("AZURE_AI_API_KEY")
    with pytest.raises(RuntimeError, match="AZURE_AI_API_KEY is not set"):
        litellm_config.resolve_model("statement-ocr")


def test_secrets_are_strippable_for_logging(config_file):
    redacted = litellm_config.redacted(litellm_config.resolve_model("statement-ocr"))
    assert "api_key" not in redacted
    assert redacted["api_base"].endswith("azure.com")


def test_an_unknown_model_is_not_an_error(config_file):
    assert litellm_config.resolve_model("no-such-model") is None


def test_no_config_anywhere_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.delenv("PDFPLAY_LITELLM_CONFIG", raising=False)
    monkeypatch.delenv("LITELLM_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    assert litellm_config.find_config() is None
    assert litellm_config.model_names() == []
    assert litellm_config.describe() == {"path": "", "models": []}


def test_a_malformed_config_does_not_take_the_parser_list_down(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    bad = tmp_path / "broken.yaml"
    bad.write_text("model_list: [ this is not: valid: yaml", encoding="utf-8")
    monkeypatch.setenv("PDFPLAY_LITELLM_CONFIG", str(bad))
    assert litellm_config.model_names() == []


def test_the_parser_offers_configured_names_and_uses_their_parameters(config_file, monkeypatch):
    pytest.importorskip("litellm")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    cls = registry.get("litellm")
    model_option = next(o for o in cls.describe()["options"] if o["name"] == "model")
    assert model_option["choices"] == ["statement-ocr", "cheap-vision"]
    assert model_option["type"] == "str", "still free text: a raw provider/model must work"

    params = cls.configured_params(cls.resolved_options({"model": "statement-ocr"}))
    assert params["model"] == "azure_ai/mistral-document-ai-2512"
    assert params["api_base"].endswith("azure.com")

    assert cls.configured_params(cls.resolved_options({"model": "anthropic/claude-sonnet-4-5"})) == {}


def test_a_configured_model_reaches_litellm_with_its_endpoint(config_file, monkeypatch):
    litellm = pytest.importorskip("litellm")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    calls: list[dict] = []

    class Response:
        _hidden_params = {"response_cost": 0.001}
        choices = [type("C", (), {"message": type("M", (), {"content": "{}"})()})()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 2})()

    monkeypatch.setattr(litellm, "completion", lambda **kw: (calls.append(kw), Response())[1])

    parser = registry.get("litellm")()
    parser.call_model(b"png", "prompt", parser.resolved_options({"model": "statement-ocr"}))

    sent = calls[0]
    assert sent["model"] == "azure_ai/mistral-document-ai-2512"
    assert sent["api_base"] == "https://my-resource.services.ai.azure.com"
    assert sent["api_key"] == "sk-azure"
    assert sent["api_version"] == "2026-01-01"


def test_an_explicit_option_still_overrides_the_config(config_file, monkeypatch):
    litellm = pytest.importorskip("litellm")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    monkeypatch.setenv("MY_KEY", "sk-override")
    calls: list[dict] = []

    class Response:
        _hidden_params: dict = {}
        choices = [type("C", (), {"message": type("M", (), {"content": "{}"})()})()]
        usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 2})()

    monkeypatch.setattr(litellm, "completion", lambda **kw: (calls.append(kw), Response())[1])

    parser = registry.get("litellm")()
    parser.call_model(
        b"png",
        "prompt",
        parser.resolved_options(
            {"model": "statement-ocr", "api_base": "http://localhost:4000", "api_key_env": "MY_KEY"}
        ),
    )
    assert calls[0]["api_base"] == "http://localhost:4000"
    assert calls[0]["api_key"] == "sk-override"
    assert calls[0]["model"] == "azure_ai/mistral-document-ai-2512", "the config still names the model"


# -- CLI --------------------------------------------------------------------


def test_the_cli_stores_and_shows_a_golden(workspace: Workspace, borderless, tmp_path, capsys):
    from pdfplay.cli import main

    meta = workspace.add_document(borderless.path)
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(GOLDEN), encoding="utf-8")

    assert main(["--workspace", str(workspace.root), "golden", meta.doc_id, "--set", str(path)]) == 0
    assert "6 field(s)" in capsys.readouterr().out

    assert main(["--workspace", str(workspace.root), "golden", meta.doc_id]) == 0
    assert json.loads(capsys.readouterr().out)["closing_balance"] == 5078.59


def test_the_cli_reports_a_missing_golden_without_traceback(workspace: Workspace, borderless, capsys):
    from pdfplay.cli import main

    meta = workspace.add_document(borderless.path)
    assert main(["--workspace", str(workspace.root), "golden", meta.doc_id]) == 1
    assert "no golden extraction" in capsys.readouterr().out
