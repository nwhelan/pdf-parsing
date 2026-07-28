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

from pdfplay import model_config
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


# -- model config.yaml ------------------------------------------------------


CONFIG = textwrap.dedent(
    """
    model_list:
      - model_name: statement-vision
        litellm_params:
          model: azure/gpt-4.1-deployment
          api_base: https://my-resource.openai.azure.com
          api_key: os.environ/AZURE_OPENAI_API_KEY
          api_version: "2026-01-01"
      - model_name: local-vision
        litellm_params:
          model: hosted_vllm/qwen2-vl
          api_base: http://localhost:8000/v1
      - model_name: statement-vision
        litellm_params:
          model: azure/a-second-deployment
    """
)


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setenv("PDFPLAY_MODEL_CONFIG", str(path))
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sk-azure")
    return path


def test_model_names_come_from_the_config(config_file):
    assert model_config.model_names() == ["statement-vision", "local-vision"]


def test_a_configured_model_brings_its_endpoint_and_key(config_file):
    resolved = model_config.resolve_model("statement-vision")
    assert resolved["base_url"] == "https://my-resource.openai.azure.com"
    assert resolved["api_version"] == "2026-01-01"
    assert resolved["api_key"] == "sk-azure", "os.environ/NAME is resolved at call time"


def test_the_provider_prefix_is_stripped_because_the_url_decides_the_provider(config_file):
    assert model_config.resolve_model("statement-vision")["model"] == "gpt-4.1-deployment"
    assert model_config.resolve_model("local-vision")["model"] == "qwen2-vl"
    assert model_config.strip_provider("gpt-4.1") == "gpt-4.1", "an unprefixed id is untouched"


def test_the_first_deployment_of_a_duplicated_name_wins(config_file):
    """A repeated model_name is proxy load balancing; here it names one endpoint."""
    assert model_config.resolve_model("statement-vision")["model"] == "gpt-4.1-deployment"


def test_an_unset_referenced_variable_says_which_one(config_file, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="AZURE_OPENAI_API_KEY is not set"):
        model_config.resolve_model("statement-vision")


def test_secrets_are_strippable_for_logging(config_file):
    redacted = model_config.redacted(model_config.resolve_model("statement-vision"))
    assert "api_key" not in redacted
    assert redacted["base_url"].endswith("azure.com")


def test_an_unknown_model_is_not_an_error(config_file):
    assert model_config.resolve_model("no-such-model") is None


def test_no_config_anywhere_is_not_an_error(monkeypatch, tmp_path):
    for name in model_config.CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    assert model_config.find_config() is None
    assert model_config.model_names() == []
    assert model_config.describe() == {"path": "", "models": []}


def test_a_malformed_config_does_not_take_the_parser_list_down(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    bad = tmp_path / "broken.yaml"
    bad.write_text("model_list: [ this is not: valid: yaml", encoding="utf-8")
    monkeypatch.setenv("PDFPLAY_MODEL_CONFIG", str(bad))
    assert model_config.model_names() == []


def test_the_parser_offers_configured_names_as_suggestions(config_file):
    pytest.importorskip("openai")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    cls = registry.get("openai-compatible")
    model_option = next(o for o in cls.describe()["options"] if o["name"] == "model")
    assert model_option["choices"] == ["statement-vision", "local-vision"]
    assert model_option["type"] == "str", "still free text: any model id must work"


def test_a_configured_model_builds_an_azure_client_pointed_at_its_endpoint(config_file):
    pytest.importorskip("openai")
    from openai import AzureOpenAI

    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    cls = registry.get("openai-compatible")
    opts = cls.resolved_options({"model": "statement-vision"})

    settings = cls.settings(opts)
    assert settings["model"] == "gpt-4.1-deployment", "the deployment name is what gets sent"
    assert settings["api_key"] == "sk-azure"

    client = cls.build_client(opts)
    assert isinstance(client, AzureOpenAI), "api_version in the config selects the Azure client"
    assert "my-resource.openai.azure.com" in str(client.base_url)


def test_a_configured_local_model_builds_a_plain_client(config_file):
    pytest.importorskip("openai")
    from openai import OpenAI

    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    cls = registry.get("openai-compatible")
    client = cls.build_client(cls.resolved_options({"model": "local-vision"}))
    assert isinstance(client, OpenAI)
    assert str(client.base_url).rstrip("/") == "http://localhost:8000/v1"


def test_an_explicit_option_still_overrides_the_config(config_file, monkeypatch):
    pytest.importorskip("openai")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    monkeypatch.setenv("MY_KEY", "sk-override")
    cls = registry.get("openai-compatible")
    settings = cls.settings(
        cls.resolved_options(
            {"model": "statement-vision", "base_url": "http://localhost:4000/v1", "api_key_env": "MY_KEY"}
        )
    )
    assert settings["base_url"] == "http://localhost:4000/v1"
    assert settings["api_key"] == "sk-override"
    assert settings["model"] == "gpt-4.1-deployment", "the config still names the model"


def test_a_model_the_config_does_not_mention_is_sent_as_written(config_file):
    pytest.importorskip("openai")
    import pdfplay.parsers  # noqa: F401
    from pdfplay import registry

    cls = registry.get("openai")
    settings = cls.settings(cls.resolved_options({"model": "gpt-4.1"}), env={"OPENAI_API_KEY": "sk-a"})
    assert settings["model"] == "gpt-4.1"
    assert settings["base_url"] == ""


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
