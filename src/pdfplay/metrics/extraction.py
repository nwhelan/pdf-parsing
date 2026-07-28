"""Score a structured extraction against a golden answer.

The bank-statement scorer works without labels by exploiting a property of the
document class. This is the other half: when you *do* have a golden set — the
right answer for a document, written once by hand — it scores any parser's
`extraction` against it, field by field.

Comparison is per leaf field rather than per document, because "the model got
the closing balance right but hallucinated the account number" is the finding,
not a single pass/fail. Values are compared the way a person would read them:

- Numbers within a tolerance, and a *string* that reads as a number is compared
  as one. Parsers disagree about whether ``"1,708.14"``, ``"$1,708.14"`` and
  ``1708.14`` are the same value; for scoring purposes they are.
- Strings ignoring case, surrounding space, and runs of whitespace.
- Dates by their parts, so ``03/01/2025`` matches ``2025-03-01``.
- ``null`` and a missing key mean the same thing: the parser did not answer.

Nothing here is specific to a document class, so it works for statements,
invoices, or whatever schema you point it at.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Currency symbols, thousands separators and trailing/leading noise around a
# number, plus the accounting convention of parentheses for negatives.
_MONEY = re.compile(r"^\(?\s*[-+]?\s*[$£€¥]?\s*[\d,\s]*\.?\d+\s*\)?$")
_DATE_PARTS = re.compile(r"\d+")
_SPACE = re.compile(r"\s+")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Reduce nested JSON to ``{dotted.path: leaf}``.

    Lists are indexed (``transactions[0].amount``) so a golden set can describe
    repeated structures without the scorer needing to know the schema.
    """
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{index}]"))
    else:
        out[prefix or "value"] = value
    return out


def as_number(value: Any) -> float | None:
    """Read a number out of a number, or out of a string that is one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not _MONEY.match(value.strip()):
        return None
    text = value.strip()
    negative = text.startswith("(") and text.endswith(")")
    cleaned = re.sub(r"[^\d.\-+]", "", text)
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if negative else number


def _date_parts(value: Any) -> set[str] | None:
    if not isinstance(value, str):
        return None
    parts = _DATE_PARTS.findall(value)
    if len(parts) < 3:
        return None
    # Compare 2025 with 25 by their last two digits, so 03/01/25 and
    # 2025-03-01 agree.
    return {p.lstrip("0")[-2:] or "0" for p in parts}


def _text(value: Any) -> str:
    return _SPACE.sub(" ", str(value)).strip().casefold()


def _plain(value: Any) -> str:
    """Text with punctuation dropped, so `A. Nwosu` and `A Nwosu` agree."""
    return _SPACE.sub(" ", "".join(c if c.isalnum() else " " for c in _text(value))).strip()


def values_match(expected: Any, actual: Any, tolerance: float = 0.011) -> bool:
    """Whether two leaf values say the same thing."""
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)

    left, right = as_number(expected), as_number(actual)
    if left is not None and right is not None:
        return math.isclose(left, right, abs_tol=tolerance)
    if left is not None or right is not None:
        # One side is a number and the other isn't; punctuation-stripping would
        # make "5078" and "5,078 USD" agree, which is not the same claim.
        return False

    if _text(expected) == _text(actual) or _plain(expected) == _plain(actual):
        return True

    left_date, right_date = _date_parts(expected), _date_parts(actual)
    if left_date and right_date:
        return left_date == right_date
    return False


def similarity(expected: Any, actual: Any) -> float:
    """How close a wrong answer was, so near-misses are visible as such."""
    import difflib

    return round(difflib.SequenceMatcher(None, _text(expected), _text(actual)).ratio(), 3)


def normalize(extraction: Any) -> Any:
    """Reduce a per-page extraction to one document-level object.

    Vision parsers ask once per page and return ``{"pages": {1: {...}}}``;
    Mistral answers for the whole document at once. Scoring them against the
    same golden set means agreeing on a shape, so pages are merged with the
    first non-null answer for a field winning — a field usually appears on one
    page, and a page that didn't see it reports null rather than nothing.
    """
    if not isinstance(extraction, dict) or set(extraction) != {"pages"}:
        return extraction
    pages = extraction["pages"]
    if not isinstance(pages, dict):
        return extraction

    merged: dict[str, Any] = {}
    lists: dict[str, list] = {}
    for _, page in sorted(pages.items(), key=lambda kv: str(kv[0])):
        if not isinstance(page, dict):
            continue
        for key, value in page.items():
            if isinstance(value, list):
                # Repeated structures (transactions, line items) accumulate
                # across pages rather than the first page winning.
                lists.setdefault(key, []).extend(value)
            elif merged.get(key) is None and value is not None:
                merged[key] = value
    merged.update(lists)
    return merged


def score_against_golden(
    extraction: Any, golden: Any, tolerance: float = 0.011
) -> dict[str, Any]:
    """Field-by-field comparison of one extraction against the golden answer.

    ``recall`` is over the golden's fields (did we get the answer?) and
    ``precision`` is over the fields the parser actually answered (of what it
    said, how much was right?) — so a parser that returns null everywhere
    scores zero recall rather than perfect precision.
    """
    predicted = flatten(normalize(extraction)) if extraction is not None else {}
    expected = flatten(golden) if golden is not None else {}

    fields: list[dict[str, Any]] = []
    correct = answered = 0

    for path, want in expected.items():
        got = predicted.get(path)
        present = path in predicted and got is not None
        ok = present and values_match(want, got, tolerance)
        correct += ok
        answered += present
        field = {
            "path": path,
            "expected": want,
            "actual": got if path in predicted else None,
            "status": "correct" if ok else ("wrong" if present else "missing"),
        }
        if field["status"] == "wrong":
            field["similarity"] = similarity(want, got)
        fields.append(field)

    for path, got in predicted.items():
        if path in expected or got is None:
            continue
        answered += 1
        fields.append({"path": path, "expected": None, "actual": got, "status": "extra"})

    n_expected = len(expected)
    return {
        "n_fields": n_expected,
        "n_correct": correct,
        "n_answered": answered,
        "accuracy": round(correct / n_expected, 4) if n_expected else None,
        "recall": round(correct / n_expected, 4) if n_expected else None,
        "precision": round(correct / answered, 4) if answered else None,
        "f1": _f1(correct, answered, n_expected),
        "fields": fields,
    }


def _f1(correct: int, answered: int, expected: int) -> float | None:
    if not answered or not expected:
        return None
    precision, recall = correct / answered, correct / expected
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)
