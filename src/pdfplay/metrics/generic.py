"""Document-class-agnostic quality signals.

None of these need ground truth. They are the first pass you run over a new
document class to see which parsers are even in the running.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any

from ..models import ParseResult
from .lines import reconstruct_lines

_WS = re.compile(r"\s+")
# U+FFFD plus C0 control characters other than tab/newline/carriage return.
_SUSPICIOUS = re.compile("[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")
_LONG_TOKEN = re.compile(r"\S{35,}")


def normalize_text(text: str) -> str:
    """NFKC + whitespace collapse, so parsers aren't penalized for spacing."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def _coverage(result: ParseResult, grid: int = 96) -> float:
    """Fraction of page area covered by boxed content, via a coarse raster mask."""
    covered = 0
    total = 0
    for page in result.pages:
        if page.width <= 0 or page.height <= 0:
            continue
        mask = bytearray(grid * grid)
        # Figures are excluded: a scanned page is one big image block, and
        # counting it would report full coverage for zero extracted text.
        boxes = [
            b.bbox
            for b in page.blocks
            if b.bbox and b.layer in ("line", "word", "region", "block") and b.kind != "figure"
        ]
        for box in boxes:
            c0 = max(0, int(box.x0 / page.width * grid))
            c1 = min(grid, int(box.x1 / page.width * grid) + 1)
            r0 = max(0, int(box.y0 / page.height * grid))
            r1 = min(grid, int(box.y1 / page.height * grid) + 1)
            for r in range(r0, r1):
                base = r * grid
                for c in range(c0, c1):
                    mask[base + c] = 1
        covered += sum(mask)
        total += grid * grid
    return covered / total if total else 0.0


def _reading_order_score(result: ParseResult) -> float:
    """Fraction of consecutive lines that move down-then-right, as emitted."""
    good = 0
    pairs = 0
    for page in result.pages:
        boxed = [b for b in page.blocks if b.layer == "line" and b.bbox]
        boxed.sort(key=lambda b: (b.order if b.order is not None else 0))
        for prev, cur in zip(boxed, boxed[1:]):
            pairs += 1
            if cur.bbox.y0 > prev.bbox.y0 - 1.0 or cur.bbox.x0 > prev.bbox.x1 - 1.0:
                good += 1
    return good / pairs if pairs else 1.0


def analyze(result: ParseResult) -> dict[str, Any]:
    """Compute quality signals for one parse result."""
    if result.status != "ok":
        return {
            "parser_id": result.parser_id,
            "status": result.status,
            "error": result.error,
            "duration_s": result.duration_s,
        }

    text = result.text()
    normalized = normalize_text(text)
    words = normalized.split()
    lines = [line for page in result.pages for line in reconstruct_lines(page)]
    line_texts = [line.text for line in lines]
    blocks = result.all_blocks()
    boxed = [b for b in blocks if b.bbox]

    layer_counts: dict[str, int] = {}
    for b in blocks:
        layer_counts[b.layer] = layer_counts.get(b.layer, 0) + 1

    confidences = [b.confidence for b in blocks if b.confidence is not None]
    duplicate = 0
    seen: set[str] = set()
    for t in line_texts:
        key = normalize_text(t)
        if key and key in seen:
            duplicate += 1
        seen.add(key)

    n_pages = max(1, len(result.pages))
    return {
        "parser_id": result.parser_id,
        "parser_name": result.parser_name,
        "status": "ok",
        "duration_s": round(result.duration_s, 3),
        "seconds_per_page": round(result.duration_s / n_pages, 3),
        "cost_usd": result.usage.cost_usd,
        "input_tokens": result.usage.input_tokens,
        "output_tokens": result.usage.output_tokens,
        "n_pages": len(result.pages),
        "n_chars": len(normalized),
        "n_words": len(words),
        "n_lines": len(line_texts),
        "n_blocks": len(blocks),
        "n_tables": len(result.all_tables()),
        "layers": layer_counts,
        "boxed_ratio": round(len(boxed) / len(blocks), 3) if blocks else 0.0,
        "page_coverage": round(_coverage(result), 3),
        "reading_order_score": round(_reading_order_score(result), 3),
        "duplicate_line_ratio": round(duplicate / len(line_texts), 3) if line_texts else 0.0,
        "suspicious_char_ratio": round(len(_SUSPICIOUS.findall(text)) / max(1, len(text)), 5),
        "run_on_tokens": len(_LONG_TOKEN.findall(normalized)),
        "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "has_markdown": bool(result.markdown),
        "warnings": len(result.warnings),
    }


def similarity(a: ParseResult, b: ParseResult) -> float:
    """Normalized text similarity in [0, 1] between two parse results."""
    ta, tb = normalize_text(a.text()), normalize_text(b.text())
    if not ta and not tb:
        return 1.0
    return difflib.SequenceMatcher(None, ta, tb).ratio()


def similarity_matrix(results: list[ParseResult]) -> dict[str, dict[str, float]]:
    ok = [r for r in results if r.status == "ok"]
    matrix: dict[str, dict[str, float]] = {}
    for a in ok:
        matrix[a.parser_id] = {}
        for b in ok:
            matrix[a.parser_id][b.parser_id] = (
                1.0 if a.parser_id == b.parser_id else round(similarity(a, b), 4)
            )
    return matrix


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance / reference length, on normalized text."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    edits = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            edits += max(i2 - i1, j2 - j1)
        elif tag in ("delete", "insert"):
            edits += (i2 - i1) + (j2 - j1)
    return round(edits / len(ref), 4)


def diff_lines(a: ParseResult, b: ParseResult, limit: int = 200) -> list[dict[str, Any]]:
    """A line-level diff between two parsers' output, for the compare view."""
    la = [line.text for page in a.pages for line in reconstruct_lines(page)]
    lb = [line.text for page in b.pages for line in reconstruct_lines(page)]
    out: list[dict[str, Any]] = []
    matcher = difflib.SequenceMatcher(None, [normalize_text(x) for x in la], [normalize_text(x) for x in lb])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        out.append(
            {
                "tag": tag,
                "left": la[i1:i2][:20],
                "right": lb[j1:j2][:20],
            }
        )
        if len(out) >= limit:
            break
    return out


def compare(a: ParseResult, b: ParseResult, limit: int = 200) -> dict[str, Any]:
    """Compare two parsers two ways: raw serialization, and row-normalized lines.

    The two often disagree in an informative way. Parsers can recover exactly
    the same content but serialize it in a different order (pdfminer walks
    layout boxes, PyMuPDF walks the content stream), which tanks raw text
    similarity while the reconstructed lines match perfectly. When that
    happens the difference is reading order, not extraction quality.
    """
    lines_a = [normalize_text(line.text) for page in a.pages for line in reconstruct_lines(page)]
    lines_b = [normalize_text(line.text) for page in b.pages for line in reconstruct_lines(page)]
    hunks = diff_lines(a, b, limit=limit)
    line_similarity = difflib.SequenceMatcher(None, lines_a, lines_b).ratio() if (lines_a or lines_b) else 1.0
    text_similarity = similarity(a, b)
    return {
        "text_similarity": round(text_similarity, 4),
        "line_similarity": round(line_similarity, 4),
        "cer": character_error_rate(a.text(), b.text()),
        "same_content_different_order": not hunks and text_similarity < 0.99,
        "hunks": hunks,
    }
