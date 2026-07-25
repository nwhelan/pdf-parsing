"""Bank-statement scoring.

The useful property of a bank statement is that it *checks itself*: every
transaction row carries a running balance, so ``balance[i] - balance[i-1]``
must equal the signed amount on row ``i``. That gives a ground-truth-free
quality score — if a parser drops a digit, merges two columns, or reorders
rows, the chain stops reconciling.

Everything here works off reconstructed lines, so it applies equally to a
text-layer parser, an OCR pass, and a vision model.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any

from ..models import ParseResult
from .generic import normalize_text
from .lines import TextLine, reconstruct_lines

# Dates: 03/14/2025, 3/14, 2025-03-14, 14 Mar 2025, Mar 14, 2025
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_DATE_PATTERNS = [
    re.compile(r"^\s*(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\b"),
    re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\b"),
    re.compile(rf"^\s*(\d{{1,2}}\s+(?:{_MONTHS})[a-z]*\.?(?:\s+\d{{2,4}})?)\b", re.I),
    re.compile(rf"^\s*((?:{_MONTHS})[a-z]*\.?\s+\d{{1,2}}(?:,?\s+\d{{2,4}})?)\b", re.I),
]

# Money: 1,234.56 / $1,234.56 / -1234.56 / (1,234.56) / 1.234,56 (EU)
_MONEY = re.compile(
    r"""(?<![\w.])
    (?P<open>\()?
    (?P<sign>[-+])?
    \s*(?P<cur>[$£€])?\s*
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\d+\.\d{2}|\d{1,3}(?:\.\d{3})+,\d{2})
    (?P<close>\))?
    (?P<trail>\s*(?:CR|DR|-|\+))?
    (?![\w])""",
    re.X | re.I,
)

_NOISE_LINE = re.compile(
    r"^(page\s+\d+|statement\s+period|account\s+(number|summary)|continued|beginning|ending)\b", re.I
)


@dataclass
class MoneyToken:
    text: str
    value: float
    x_center: float | None = None


@dataclass
class Transaction:
    page: int
    line_index: int
    date: str
    description: str
    amount: float | None
    balance: float | None
    raw: str
    amount_x: float | None = None
    balance_x: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BankStatementReport:
    n_lines: int = 0
    n_transactions: int = 0
    n_with_amount: int = 0
    n_with_balance: int = 0
    reconciled_pairs: int = 0
    reconcilable_pairs: int = 0
    reconciliation_rate: float = 0.0
    amount_column_consistency: float | None = None
    amount_columns_detected: int = 0
    balance_column_consistency: float | None = None
    date_formats: dict[str, int] = field(default_factory=dict)
    opening_balance: float | None = None
    closing_balance: float | None = None
    sum_of_amounts: float | None = None
    closing_minus_opening: float | None = None
    totals_match: bool | None = None
    breaks: list[dict[str, Any]] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_money(text: str) -> list[MoneyToken]:
    """Find money-shaped tokens in a line, with sign conventions applied."""
    out: list[MoneyToken] = []
    for m in _MONEY.finditer(text):
        raw_num = m.group("num")
        if "," in raw_num and re.search(r"\.\d{3}", raw_num):  # 1.234,56
            num = raw_num.replace(".", "").replace(",", ".")
        else:
            num = raw_num.replace(",", "").replace(" ", "")
        try:
            value = float(num)
        except ValueError:
            continue
        trail = (m.group("trail") or "").strip().upper()
        negative = bool(m.group("open") and m.group("close"))
        negative = negative or m.group("sign") == "-"
        negative = negative or trail in ("-", "DR")
        out.append(MoneyToken(text=m.group(0).strip(), value=-value if negative else value))
    return out


def _match_date(text: str) -> tuple[str, str] | None:
    for i, pattern in enumerate(_DATE_PATTERNS):
        m = pattern.match(text)
        if m:
            return m.group(1).strip(), f"pattern_{i}"
    return None


def _token_x(line: TextLine, needle: str) -> float | None:
    needle = needle.strip().lstrip("$£€").strip()
    for text, bbox in line.tokens:
        if needle and needle in text:
            return bbox.cx
    return None


def extract_transactions(result: ParseResult) -> list[Transaction]:
    """Pull transaction-looking rows out of a parse result."""
    transactions: list[Transaction] = []
    index = 0
    for page in result.pages:
        for line in reconstruct_lines(page):
            index += 1
            text = normalize_text(line.text)
            if not text or _NOISE_LINE.match(text):
                continue
            date_match = _match_date(text)
            if not date_match:
                continue
            date, _fmt = date_match
            money = parse_money(text[len(date) :])
            amount = balance = None
            amount_x = balance_x = None
            if len(money) >= 2:
                amount, balance = money[-2].value, money[-1].value
                amount_x = _token_x(line, money[-2].text)
                balance_x = _token_x(line, money[-1].text)
            elif len(money) == 1:
                amount = money[0].value
                amount_x = _token_x(line, money[0].text)

            description = text[len(date) :]
            for token in money:
                description = description.replace(token.text, " ")
            transactions.append(
                Transaction(
                    page=page.page_number,
                    line_index=index,
                    date=date,
                    description=normalize_text(description),
                    amount=amount,
                    balance=balance,
                    raw=text,
                    amount_x=amount_x,
                    balance_x=balance_x,
                )
            )
    return transactions


def _column_consistency(values: list[float | None], gap: float = 20.0) -> tuple[float | None, int]:
    """How tightly a numeric column holds its x position.

    Statements often have *two* amount columns (withdrawals and deposits), so a
    naive standard deviation would score a perfectly-parsed statement badly.
    Split on the largest gap when it exceeds ``gap`` points, then score the
    within-cluster spread. Returns ``(score, n_columns)``; 1.0 is perfectly
    aligned.
    """
    xs = sorted(v for v in values if v is not None)
    if len(xs) < 3:
        return None, 0

    clusters = [xs]
    gaps = [(b - a, i) for i, (a, b) in enumerate(zip(xs, xs[1:]))]
    if gaps:
        widest, at = max(gaps)
        if widest > gap:
            clusters = [xs[: at + 1], xs[at + 1 :]]

    total = 0.0
    weight = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        total += statistics.pstdev(cluster) * len(cluster)
        weight += len(cluster)
    spread = total / weight if weight else 0.0
    return round(max(0.0, 1.0 - spread / 40.0), 3), len(clusters)


def analyze(result: ParseResult, tolerance: float = 0.011) -> BankStatementReport:
    """Score how well a parser preserved a bank statement's ledger structure."""
    report = BankStatementReport()
    if result.status != "ok":
        return report

    lines = [line for page in result.pages for line in reconstruct_lines(page)]
    report.n_lines = len(lines)

    transactions = extract_transactions(result)
    report.n_transactions = len(transactions)
    report.n_with_amount = sum(1 for t in transactions if t.amount is not None)
    report.n_with_balance = sum(1 for t in transactions if t.balance is not None)
    report.transactions = [t.as_dict() for t in transactions]

    for t in transactions:
        match = _match_date(t.raw)
        if match:
            report.date_formats[match[1]] = report.date_formats.get(match[1], 0) + 1

    report.amount_column_consistency, report.amount_columns_detected = _column_consistency(
        [t.amount_x for t in transactions]
    )
    report.balance_column_consistency, _ = _column_consistency([t.balance_x for t in transactions])

    # Running-balance reconciliation: the ground-truth-free signal.
    with_balance = [t for t in transactions if t.balance is not None and t.amount is not None]
    for prev, cur in zip(with_balance, with_balance[1:]):
        report.reconcilable_pairs += 1
        delta = cur.balance - prev.balance
        # A parser can't always recover the debit/credit sign, so accept either.
        if abs(delta - cur.amount) <= tolerance or abs(delta + cur.amount) <= tolerance:
            report.reconciled_pairs += 1
        else:
            report.breaks.append(
                {
                    "line_index": cur.line_index,
                    "page": cur.page,
                    "expected_delta": round(delta, 2),
                    "amount": cur.amount,
                    "raw": cur.raw,
                }
            )
    report.reconciliation_rate = (
        round(report.reconciled_pairs / report.reconcilable_pairs, 3)
        if report.reconcilable_pairs
        else 0.0
    )

    if with_balance:
        report.opening_balance = with_balance[0].balance
        report.closing_balance = with_balance[-1].balance
        report.closing_minus_opening = round(report.closing_balance - report.opening_balance, 2)
        signed_sum = 0.0
        for prev, cur in zip(with_balance, with_balance[1:]):
            signed_sum += cur.balance - prev.balance
        report.sum_of_amounts = round(signed_sum, 2)
        report.totals_match = abs(signed_sum - report.closing_minus_opening) <= tolerance

    return report


# -- ground-truth comparison ------------------------------------------------


def score_against_ledger(
    result: ParseResult, ledger: list[dict[str, Any]], tolerance: float = 0.011
) -> dict[str, Any]:
    """Precision/recall of extracted transactions against a known ledger.

    A predicted row matches a ledger row when the amounts agree within
    ``tolerance`` and the dates share a normalized form. Descriptions are
    scored separately so a parser isn't punished twice for one bad row.
    """
    import difflib

    predicted = extract_transactions(result)
    remaining = list(enumerate(ledger))
    matches: list[tuple[Transaction, dict[str, Any]]] = []

    for p in predicted:
        best_idx = None
        for pos, (_, truth) in enumerate(remaining):
            truth_amount = float(truth.get("amount", 0.0))
            if p.amount is None:
                continue
            if abs(abs(p.amount) - abs(truth_amount)) > tolerance:
                continue
            if not _dates_agree(p.date, str(truth.get("date", ""))):
                continue
            best_idx = pos
            break
        if best_idx is not None:
            _, truth = remaining.pop(best_idx)
            matches.append((p, truth))

    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ledger) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    desc_scores = [
        difflib.SequenceMatcher(
            None, normalize_text(p.description).lower(), normalize_text(str(t.get("description", ""))).lower()
        ).ratio()
        for p, t in matches
    ]
    balance_hits = sum(
        1
        for p, t in matches
        if p.balance is not None
        and t.get("balance") is not None
        and abs(p.balance - float(t["balance"])) <= tolerance
    )

    return {
        "parser_id": result.parser_id,
        "n_truth": len(ledger),
        "n_predicted": len(predicted),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "description_similarity": round(sum(desc_scores) / len(desc_scores), 3) if desc_scores else None,
        "balance_accuracy": round(balance_hits / tp, 3) if tp else 0.0,
        "missed": [truth for _, truth in remaining][:20],
    }


_DATE_SPLIT = re.compile(r"[^0-9a-z]+", re.I)
_MONTH_NUM = {
    "jan": "1", "feb": "2", "mar": "3", "apr": "4", "may": "5", "jun": "6",
    "jul": "7", "aug": "8", "sep": "9", "oct": "10", "nov": "11", "dec": "12",
}


def _date_parts(value: str) -> set[str]:
    """Split a date into comparable parts, mapping month names onto numbers."""
    parts = set()
    for piece in _DATE_SPLIT.split(value.lower()):
        if not piece:
            continue
        if piece.isdigit():
            parts.add(str(int(piece)))
        else:
            parts.add(_MONTH_NUM.get(piece[:3], piece[:3]))
    return parts


def _dates_agree(a: str, b: str) -> bool:
    """Loose date equality: same day/month parts, tolerating a missing year."""
    if not a or not b:
        return False
    sa, sb = _date_parts(a), _date_parts(b)
    shared = sa & sb
    return len(shared) >= min(len(sa), len(sb), 2)
