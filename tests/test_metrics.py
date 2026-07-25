from pdfplay.metrics import bank_statement, generic
from pdfplay.metrics.lines import TextLine, merge_rows
from pdfplay.models import BBox, Block, PageResult, ParseResult


def _line(text: str, x0: float, y0: float, x1: float, y1: float) -> TextLine:
    return TextLine(page=1, text=text, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1), source="line")


def test_merge_rows_joins_cells_on_the_same_baseline():
    # MuPDF emits one "line" per table cell when the gaps are wide.
    fragments = [
        _line("03/01/2025", 54, 200, 96, 211),
        _line("CHECK 1042", 116, 200, 167, 211),
        _line("31.61", 394, 200, 416, 211),
        _line("6,492.77", 524, 200, 558, 211),
        _line("03/02/2025", 54, 217, 96, 228),
    ]
    merged = merge_rows(fragments)
    assert len(merged) == 2
    assert merged[0].text == "03/01/2025 CHECK 1042 31.61 6,492.77"
    assert merged[0].source.endswith("+row")
    assert merged[1].text == "03/02/2025"


def test_merge_rows_keeps_separate_rows_apart():
    merged = merge_rows([_line("a", 0, 0, 10, 10), _line("b", 0, 40, 10, 50)])
    assert [m.text for m in merged] == ["a", "b"]


def test_money_parsing_handles_common_formats():
    values = [t.value for t in bank_statement.parse_money("1,234.56 (78.90) -12.00 45.00CR 3.00DR")]
    assert values == [1234.56, -78.90, -12.00, 45.00, -3.00]


def test_money_parser_does_not_swallow_a_preceding_store_number():
    # Regression: "#445 240.29" must not parse as 445240.29 via a space-separated
    # thousands group.
    tokens = bank_statement.parse_money("TRADER JOES #445 240.29 5,912.00")
    assert [t.value for t in tokens] == [240.29, 5912.00]


def _result_from_lines(texts: list[str]) -> ParseResult:
    blocks = [
        Block(
            id=f"l{i}",
            page=1,
            layer="line",
            text=text,
            bbox=BBox(x0=54, y0=100 + i * 20, x1=558, y1=112 + i * 20),
            order=i,
        )
        for i, text in enumerate(texts)
    ]
    page = PageResult(page_number=1, width=612, height=792, blocks=blocks, text="\n".join(texts))
    return ParseResult(parser_id="fake", status="ok", pages=[page])


def test_reconciliation_rate_is_perfect_on_a_consistent_ledger():
    result = _result_from_lines(
        [
            "03/01/2025 OPENING PURCHASE 100.00 900.00",
            "03/02/2025 COFFEE 10.00 890.00",
            "03/03/2025 PAYCHECK 500.00 1,390.00",
        ]
    )
    report = bank_statement.analyze(result)
    assert report.n_transactions == 3
    assert report.reconciliation_rate == 1.0
    assert report.breaks == []
    assert report.totals_match is True


def test_reconciliation_catches_a_corrupted_amount():
    result = _result_from_lines(
        [
            "03/01/2025 OPENING PURCHASE 100.00 900.00",
            "03/02/2025 COFFEE 1.00 890.00",  # amount lost a digit
            "03/03/2025 PAYCHECK 500.00 1,390.00",
        ]
    )
    report = bank_statement.analyze(result)
    assert report.reconciliation_rate < 1.0
    assert len(report.breaks) == 1
    assert report.breaks[0]["expected_delta"] == -10.0


def test_ledger_scoring_reports_a_dropped_row():
    result = _result_from_lines(
        [
            "03/01/2025 COFFEE 10.00 890.00",
            "03/03/2025 PAYCHECK 500.00 1,390.00",
        ]
    )
    ledger = [
        {"date": "2025-03-01", "description": "COFFEE", "amount": -10.0, "balance": 890.0},
        {"date": "2025-03-02", "description": "BOOKS", "amount": -25.0, "balance": 865.0},
        {"date": "2025-03-03", "description": "PAYCHECK", "amount": 500.0, "balance": 1390.0},
    ]
    score = bank_statement.score_against_ledger(result, ledger)
    assert score["true_positives"] == 2
    assert score["false_negatives"] == 1
    assert score["recall"] < 1.0
    assert score["precision"] == 1.0
    assert score["missed"][0]["description"] == "BOOKS"


def test_markdown_table_rows_are_read_as_transactions():
    """A parser that emits a real Markdown table must not be penalized for it."""
    result = _result_from_lines(
        [
            "|Date|Description|Withdrawals|Deposits|Balance|",
            "|---|---|---|---|---|",
            "|03/01/2025|OPENING PURCHASE|100.00||900.00|",
            "|03/02/2025|COFFEE<br>SHOP|10.00||890.00|",
            "|03/03/2025|PAYCHECK||500.00|1,390.00|",
        ]
    )
    report = bank_statement.analyze(result)
    assert report.n_transactions == 3
    assert report.reconciliation_rate == 1.0
    assert report.transactions[1]["description"] == "COFFEE SHOP"


def test_flatten_row_leaves_plain_lines_untouched():
    assert bank_statement.flatten_row("03/01/2025 COFFEE 10.00 890.00") == "03/01/2025 COFFEE 10.00 890.00"
    assert bank_statement.flatten_row("|---|---|") == ""


def test_character_error_rate_is_zero_for_whitespace_differences():
    assert generic.character_error_rate("hello  world", "hello world") == 0.0
    assert generic.character_error_rate("hello world", "hella world") > 0.0


def test_coverage_ignores_full_page_figures():
    # A scanned page is one big image block; it must not read as full coverage.
    page = PageResult(
        page_number=1,
        width=100,
        height=100,
        blocks=[Block(id="f", page=1, layer="block", kind="figure", bbox=BBox(x0=0, y0=0, x1=100, y1=100))],
    )
    result = ParseResult(parser_id="fake", status="ok", pages=[page])
    assert generic.analyze(result)["page_coverage"] == 0.0
