"""Contract tests: every available parser must satisfy the normalized model."""

from __future__ import annotations

import pytest

import pdfplay.parsers  # noqa: F401  (importing it registers the adapters)
from pdfplay import registry
from pdfplay.geometry import page_geometry
from pdfplay.metrics import bank_statement, generic

LOCAL_TEXT_PARSERS = ["pymupdf", "pdfplumber", "pypdfium2", "pdfminer"]
# Parsers that must satisfy the box/page contract, including the layout model.
BOXED_PARSERS = [*LOCAL_TEXT_PARSERS, "pymupdf-layout"]


def _available(parser_id: str):
    cls = registry.get(parser_id)
    avail = cls.check_availability()
    if not avail.available:
        pytest.skip(f"{parser_id} unavailable: {avail.reason}")
    return cls


@pytest.mark.parametrize("parser_id", BOXED_PARSERS)
def test_parser_returns_pages_matching_the_pdf(parser_id, borderless):
    cls = _available(parser_id)
    parsed = cls().parse(borderless.path, None, cls.resolved_options({}))
    geometry = page_geometry(borderless.path)

    assert len(parsed.pages) == len(geometry)
    for page in parsed.pages:
        geo = geometry[page.page_number]
        assert abs(page.width - geo.width) < 0.5
        assert abs(page.height - geo.height) < 0.5


@pytest.mark.parametrize("parser_id", BOXED_PARSERS)
def test_boxes_are_finite_and_inside_the_page(parser_id, borderless):
    cls = _available(parser_id)
    parsed = cls().parse(borderless.path, None, cls.resolved_options({}))
    for page in parsed.pages:
        for block in page.blocks:
            if block.bbox is None:
                continue
            assert block.bbox.is_finite()
            assert block.bbox.x0 <= block.bbox.x1
            assert block.bbox.y0 <= block.bbox.y1
            # 2pt of slack for glyph overhang.
            assert -2 <= block.bbox.x0 and block.bbox.x1 <= page.width + 2
            assert -2 <= block.bbox.y0 and block.bbox.y1 <= page.height + 2


@pytest.mark.parametrize("parser_id", BOXED_PARSERS)
def test_boxes_are_top_left_origin(parser_id, borderless):
    """The bank name is at the top of page 1, so its box must have a small y."""
    cls = _available(parser_id)
    parsed = cls().parse(borderless.path, [1], cls.resolved_options({}))
    page = parsed.pages[0]
    hits = [b for b in page.blocks if b.bbox and "MERIDIAN" in b.text]
    assert hits, "expected the bank name on page 1"
    assert min(b.bbox.y0 for b in hits) < page.height / 3


@pytest.mark.parametrize("parser_id", BOXED_PARSERS)
def test_page_selection_is_honoured(parser_id, borderless):
    assert len(page_geometry(borderless.path)) >= 2, "fixture must span multiple pages"
    cls = _available(parser_id)
    parsed = cls().parse(borderless.path, [2], cls.resolved_options({}))
    assert [p.page_number for p in parsed.pages] == [2]


@pytest.mark.parametrize("parser_id", BOXED_PARSERS)
def test_statement_ledger_is_recovered(parser_id, borderless):
    """Every text-layer parser should read a clean statement essentially perfectly."""
    cls = _available(parser_id)
    parsed = cls().parse(borderless.path, None, cls.resolved_options({}))
    from pdfplay.models import ParseResult

    result = ParseResult(parser_id=parser_id, status="ok", pages=parsed.pages)
    report = bank_statement.analyze(result)
    truth = borderless.ledger["transactions"]

    assert report.n_transactions == len(truth)
    assert report.reconciliation_rate == 1.0
    score = bank_statement.score_against_ledger(result, truth)
    assert score["f1"] == 1.0


@pytest.mark.parametrize("parser_id", LOCAL_TEXT_PARSERS)
def test_text_layer_parsers_find_nothing_in_a_scan(parser_id, scanned):
    """The scanned variant has no text layer — parsers must not hallucinate one."""
    cls = _available(parser_id)
    parsed = cls().parse(scanned.path, None, cls.resolved_options({}))
    text = "".join(p.text for p in parsed.pages)
    assert generic.normalize_text(text) == ""


def test_tesseract_reads_the_scan_when_installed(scanned):
    cls = _available("tesseract")
    parsed = cls().parse(scanned.path, [1], cls.resolved_options({"dpi": 200}))
    from pdfplay.models import ParseResult

    result = ParseResult(parser_id="tesseract", status="ok", pages=parsed.pages)
    report = bank_statement.analyze(result)
    assert report.n_transactions > 5
    assert report.reconciliation_rate > 0.9


def test_pymupdf_layout_reads_a_scan_with_its_ocr_fallback(scanned):
    """The layout engine OCRs a page with no text layer — unlike its own classic engine."""
    cls = _available("pymupdf-layout")
    from pdfplay.models import ParseResult

    parsed = cls().parse(scanned.path, [1], cls.resolved_options({"engine": "layout"}))
    result = ParseResult(parser_id="pymupdf-layout", status="ok", pages=parsed.pages)
    report = bank_statement.analyze(result)
    assert report.n_transactions > 5
    assert report.reconciliation_rate > 0.9

    classic = cls().parse(scanned.path, [1], cls.resolved_options({"engine": "classic"}))
    assert generic.normalize_text("".join(p.text for p in classic.pages)) == ""


def test_pymupdf_layout_classifies_regions_and_rebuilds_the_table(borderless):
    cls = _available("pymupdf-layout")
    parsed = cls().parse(borderless.path, None, cls.resolved_options({"engine": "layout"}))

    kinds = {b.kind for page in parsed.pages for b in page.blocks}
    assert {"header", "title", "table", "footer"} <= kinds
    assert all(b.layer == "region" for page in parsed.pages for b in page.blocks)

    tables = [t for page in parsed.pages for t in page.tables]
    assert tables, "the layout engine should emit a Markdown table for the ledger"
    assert sum(t.n_rows for t in tables) >= len(borderless.ledger["transactions"])
    assert all(t.n_cols == 5 for t in tables)
    assert parsed.markdown and "|---|" in parsed.markdown


def test_pymupdf_layout_classic_engine_gives_word_boxes(borderless):
    cls = _available("pymupdf-layout")
    parsed = cls().parse(borderless.path, [1], cls.resolved_options({"engine": "classic"}))
    words = [b for b in parsed.pages[0].blocks if b.layer == "word"]
    assert len(words) > 50
    assert all(b.bbox is not None for b in words)


def test_registry_reports_every_parser_with_a_reason():
    described = registry.describe_all()
    assert {d["id"] for d in described} >= set(BOXED_PARSERS)
    for spec in described:
        assert spec["available"] or spec["unavailable_reason"], spec["id"]
        assert spec["description"]
