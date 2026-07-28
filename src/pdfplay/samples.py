"""Synthetic sample documents with known ground truth.

Real bank statements can't be checked into a repo, and without labels you
can't tell a parser that reads a statement correctly from one that reads it
plausibly. These generators produce statements whose ledger we already know,
in the three variants that actually separate parsers:

``ruled``       — a table with ruling lines (the easy case)
``borderless``  — same content, no rules, alignment only (the common case)
``scanned``     — the borderless page rasterized and slightly skewed (no text layer)
"""

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAGE_W, PAGE_H = 612.0, 792.0
MARGIN = 54.0

MERCHANTS = [
    ("SQ *BLUE BOTTLE COFFEE", "debit"),
    ("AMZN MKTP US*2K4LZ", "debit"),
    ("SHELL OIL 57442136", "debit"),
    ("TRADER JOES #445", "debit"),
    ("NETFLIX.COM", "debit"),
    ("PG&E WEB ONLINE", "debit"),
    ("DIRECT DEP ACME CORP PAYROLL", "credit"),
    ("ZELLE FROM J RIVERA", "credit"),
    ("ATM WITHDRAWAL 1425 MARKET ST", "debit"),
    ("CHECK 1042", "debit"),
    ("TRANSFER TO SAVINGS ...4417", "debit"),
    ("INTEREST PAYMENT", "credit"),
    ("MONTHLY MAINTENANCE FEE", "debit"),
    ("REFUND UNITED AIRLINES", "credit"),
]

COLUMNS = [
    # (label, x, width, alignment) — laid out to fit inside the page margins.
    ("Date", MARGIN, 58.0, "left"),
    ("Description", MARGIN + 62, 228.0, "left"),
    ("Withdrawals", MARGIN + 294, 68.0, "right"),
    ("Deposits", MARGIN + 366, 68.0, "right"),
    ("Balance", MARGIN + 438, 66.0, "right"),
]


@dataclass
class SampleDocument:
    path: Path
    ledger: dict[str, Any]

    def write_ledger(self, path: Path | None = None) -> Path:
        target = path or self.path.with_suffix(".ledger.json")
        target.write_text(json.dumps(self.ledger, indent=2, ensure_ascii=False), encoding="utf-8")
        return target


def _build_ledger(seed: int, n: int) -> dict[str, Any]:
    rng = random.Random(seed)
    balance = round(rng.uniform(1200, 8000), 2)
    opening = balance
    transactions: list[dict[str, Any]] = []
    day = 1
    for _ in range(n):
        # Several transactions can land on the same day, as in a real statement.
        day += rng.choice([0, 0, 1, 1, 1, 2])
        if day > 31:
            break
        description, direction = rng.choice(MERCHANTS)
        if direction == "credit":
            amount = round(rng.uniform(45, 2600), 2)
        else:
            amount = -round(rng.uniform(4, 480), 2)
        balance = round(balance + amount, 2)
        transactions.append(
            {
                "date": f"03/{day:02d}/2025",
                "description": description,
                "amount": amount,
                "balance": balance,
            }
        )
    return {
        "doc_class": "bank_statement",
        "account_name": "MERIDIAN COMMUNITY BANK",
        "account_number": "****4417",
        "period": "March 1, 2025 - March 31, 2025",
        "opening_balance": opening,
        "closing_balance": balance,
        "transactions": transactions,
    }


def _money(value: float) -> str:
    return f"{abs(value):,.2f}"


def _draw_statement(ledger: dict[str, Any], ruled: bool) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    rows_per_page = 26
    transactions = ledger["transactions"]
    chunks = [transactions[i : i + rows_per_page] for i in range(0, len(transactions), rows_per_page)] or [[]]

    for page_index, chunk in enumerate(chunks):
        page = doc.new_page(width=PAGE_W, height=PAGE_H)
        y = MARGIN

        page.insert_text((MARGIN, y + 10), ledger["account_name"], fontname="hebo", fontsize=15)
        page.insert_text(
            (MARGIN, y + 26), "Statement of Account", fontname="helv", fontsize=10, color=(0.3, 0.3, 0.3)
        )
        page.insert_textbox(
            pymupdf.Rect(PAGE_W - MARGIN - 220, y, PAGE_W - MARGIN, y + 60),
            f"Account {ledger['account_number']}\n{ledger['period']}\nPage {page_index + 1} of {len(chunks)}",
            fontname="helv",
            fontsize=9,
            align=pymupdf.TEXT_ALIGN_RIGHT,
        )
        y += 60

        if page_index == 0:
            page.draw_rect(
                pymupdf.Rect(MARGIN, y, PAGE_W - MARGIN, y + 46),
                color=(0.75, 0.78, 0.82),
                fill=(0.96, 0.97, 0.98),
                width=0.6,
            )
            page.insert_text((MARGIN + 10, y + 17), "Account Summary", fontname="hebo", fontsize=9)
            page.insert_text(
                (MARGIN + 10, y + 33),
                f"Beginning balance  {_money(ledger['opening_balance'])}"
                f"        Ending balance  {_money(ledger['closing_balance'])}",
                fontname="helv",
                fontsize=9,
            )
            y += 62

        # Column headers
        for label, x, width, align in COLUMNS:
            rect = pymupdf.Rect(x, y, x + width, y + 16)
            page.insert_textbox(
                rect,
                label,
                fontname="hebo",
                fontsize=8.5,
                align=pymupdf.TEXT_ALIGN_RIGHT if align == "right" else pymupdf.TEXT_ALIGN_LEFT,
            )
        y += 18
        page.draw_line(pymupdf.Point(MARGIN, y), pymupdf.Point(PAGE_W - MARGIN, y), width=0.8)
        y += 6

        row_h = 17.0
        table_top = y
        for row in chunk:
            values = [
                row["date"],
                row["description"],
                _money(row["amount"]) if row["amount"] < 0 else "",
                _money(row["amount"]) if row["amount"] > 0 else "",
                _money(row["balance"]),
            ]
            for (label, x, width, align), value in zip(COLUMNS, values):
                if not value:
                    continue
                page.insert_textbox(
                    pymupdf.Rect(x, y, x + width, y + row_h),
                    value,
                    fontname="helv",
                    fontsize=8.5,
                    align=pymupdf.TEXT_ALIGN_RIGHT if align == "right" else pymupdf.TEXT_ALIGN_LEFT,
                )
            if ruled:
                page.draw_line(
                    pymupdf.Point(MARGIN, y + row_h - 2),
                    pymupdf.Point(PAGE_W - MARGIN, y + row_h - 2),
                    color=(0.82, 0.84, 0.86),
                    width=0.4,
                )
            y += row_h

        if ruled:
            page.draw_rect(
                pymupdf.Rect(MARGIN, table_top - 6, PAGE_W - MARGIN, y - 2),
                color=(0.6, 0.63, 0.68),
                width=0.8,
            )
            for _, x, width, align in COLUMNS[1:]:
                edge = x - 6 if align == "left" else x - 6
                page.draw_line(
                    pymupdf.Point(edge, table_top - 6),
                    pymupdf.Point(edge, y - 2),
                    color=(0.75, 0.78, 0.82),
                    width=0.4,
                )

        page.insert_textbox(
            pymupdf.Rect(MARGIN, PAGE_H - MARGIN - 24, PAGE_W - MARGIN, PAGE_H - MARGIN),
            "Member FDIC. Questions? Call 1-800-555-0142. "
            "Please report any discrepancy within 60 days of the statement date.",
            fontname="helv",
            fontsize=7,
            color=(0.4, 0.4, 0.4),
        )

    data = doc.tobytes()
    doc.close()
    return data


def _rasterize(pdf_bytes: bytes, dpi: int = 150, skew_deg: float = 0.4, noise: int = 8) -> bytes:
    """Turn a text PDF into an image-only PDF that looks like a desk scan."""
    import pymupdf
    from PIL import Image, ImageFilter

    src = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    out = pymupdf.open()
    rng = random.Random(7)
    zoom = dpi / 72.0
    for page in src:
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        image = image.rotate(skew_deg, resample=Image.BICUBIC, fillcolor=245, expand=False)
        image = image.filter(ImageFilter.GaussianBlur(0.4))
        if noise:
            pixels = image.load()
            w, h = image.size
            for _ in range((w * h) // 400):
                x, y = rng.randrange(w), rng.randrange(h)
                pixels[x, y] = max(0, min(255, pixels[x, y] + rng.randint(-noise * 6, noise * 2)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=72)

        new_page = out.new_page(width=page.rect.width, height=page.rect.height)
        new_page.insert_image(new_page.rect, stream=buf.getvalue())
    data = out.tobytes()
    src.close()
    out.close()
    return data


def make_bank_statement(
    out_path: Path | str,
    variant: str = "ruled",
    seed: int = 7,
    n_transactions: int = 34,
) -> SampleDocument:
    """Generate a synthetic bank statement plus its ledger.

    ``variant`` is one of ``ruled``, ``borderless``, or ``scanned``.
    """
    if variant not in ("ruled", "borderless", "scanned"):
        raise ValueError(f"unknown variant: {variant}")

    ledger = _build_ledger(seed, n_transactions)
    pdf_bytes = _draw_statement(ledger, ruled=(variant == "ruled"))
    if variant == "scanned":
        pdf_bytes = _rasterize(pdf_bytes)

    ledger["variant"] = variant
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)
    return SampleDocument(path=out_path, ledger=ledger)


def make_all(out_dir: Path | str = "samples", seed: int = 7) -> list[SampleDocument]:
    out_dir = Path(out_dir)
    docs = []
    for variant in ("ruled", "borderless", "scanned"):
        doc = make_bank_statement(out_dir / f"bank_statement_{variant}.pdf", variant=variant, seed=seed)
        doc.write_ledger()
        docs.append(doc)
    return docs
