# pdfplay

A playground for running the same PDF through several parsers and comparing what
each one actually saw — rendered page on the left with bounding boxes drawn on
top, extracted text and scores on the right.

The point is not "which parser is best" in the abstract. It's to find which
parser is best **for a document class you care about** (bank statements, first),
and then build an extraction pipeline around the winners.

```
┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐
│ workspace       │   │ parser adapters  │   │ scoring            │
│  docs + renders │──▶│ local + remote   │──▶│ generic signals    │
│  cached results │   │ (same interface) │   │ bank-statement     │
└─────────────────┘   └──────────────────┘   └────────────────────┘
                              │                        │
                              └──────▶ web viewer ◀────┘
```

## Quick start

```bash
uv venv && uv pip install -e .          # or: python -m venv .venv && pip install -e .
pdfplay samples                          # generate synthetic statements with known ledgers
pdfplay add samples/bank_statement_borderless.pdf --class bank_statement
pdfplay run <doc_id> --all               # run every available parser
pdfplay score <doc_id>                   # generic + bank-statement scores as a table
pdfplay serve                            # http://127.0.0.1:8000 for the visual comparison
```

`pdfplay list` shows which parsers are installed and which are one `pip install`
away.

## What's included

| Parser | Kind | Notes |
|---|---|---|
| `pymupdf` | local | MuPDF text layer, word/line/block boxes, geometric table finder. Fastest. |
| `pdfplumber` | local | pdfminer + word clustering. Its `text` table strategy handles **borderless** tables. |
| `pypdfium2` | local | Chrome's PDF engine. Text rects only — a good "what's really in the text layer" baseline. |
| `pdfminer` | local | Tunable `LAParams`; shows how sensitive paragraph grouping is on a document class. |
| `docling` | local (extra) | Layout model + TableFormer + optional OCR. Labelled regions, Markdown export. |
| `unstructured` | local (extra) | Element classification, `hi_res` layout model. |
| `tesseract` | local (extra) | Rasterize + OCR, ignoring the text layer. The scanned-document control. |
| `claude` | remote (extra) | Page image → structured JSON transcription, via Anthropic structured outputs. |
| `openai` | remote (extra) | Same contract via Chat Completions `json_schema`. |
| `gemini` | remote (extra) | Same contract, using Gemini's native `[ymin,xmin,ymax,xmax]` box convention. |

Install the optional ones as needed:

```bash
uv pip install -e '.[docling]'      # also: unstructured, ocr, anthropic, openai, gemini, vision
```

Remote parsers read their credentials from the environment
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`) and report themselves
as unavailable — with the reason — when a key is missing. Nothing is ever sent
anywhere unless you explicitly run a remote parser.

## The normalized model

Every adapter returns the same shape, so scoring and the viewer don't care
which library produced the output:

- **Coordinates** are PDF points, top-left origin, y down. Parsers that use a
  bottom-left origin convert on the way in. The page carries its own
  width/height in points, so the viewer needs one scale factor to overlay boxes
  on a render.
- **Layers** are granularity bands: `word`, `line`, `block`, `region`, `table`,
  `cell`. A parser emits whichever it can, and the viewer lets you toggle
  between them — this is where the differences actually show up.
- **Kinds** are semantic labels: `text`, `title`, `header`, `footer`, `list`,
  `table`, `figure`, `key_value`, `caption`, `formula`. Layout-model parsers
  fill these in; text-layer parsers mostly don't.

One normalization is applied before scoring: **line fragments on the same
visual row are merged**. Parsers disagree about what a "line" is — given a
table row with wide column gaps, MuPDF emits one line *per cell* while
pdfplumber emits one line for the whole row. Neither is wrong, but comparing
them requires agreeing on a unit, and for row-structured documents the row is
it. The merge is applied identically to every parser, so it normalizes the
comparison rather than favouring one library. The overlay always shows each
parser's *own* raw boxes.

Results are cached on disk keyed by `(parser, options, page selection)`, so
re-running a comparison is instant and a remote model is never billed twice for
the same request.

## Scoring

**Generic signals** (`pdfplay.metrics.generic`) need no ground truth: character
and word counts, page coverage, reading-order monotonicity, duplicate lines,
suspicious characters, run-on tokens, time per page, and token cost. Plus a
cross-parser text similarity matrix — when four parsers agree and one doesn't,
that's usually the story.

**Bank statements** (`pdfplay.metrics.bank_statement`) get a domain scorer that
exploits a property of the document class: a statement checks itself. Each row
carries a running balance, so `balance[i] - balance[i-1]` must equal the signed
amount on row `i`. The scorer extracts transaction rows, then reports:

- `reconciliation_rate` — fraction of consecutive rows whose balance delta
  matches the row's amount. **This is a quality score with no labels at all.** A
  parser that drops a digit, merges the withdrawal and deposit columns, or
  reorders rows will fail it.
- `amount_column_consistency` / `balance_column_consistency` — how tightly the
  numeric columns stay aligned in x, i.e. whether column structure survived.
- `totals_match`, `opening_balance`, `closing_balance`, and the specific rows
  where the chain broke.

When you *do* have labels, `score_against_ledger()` gives precision / recall /
F1 against a known ledger, matching on amount + date and scoring descriptions
separately so one bad row isn't punished twice.

The synthetic samples ship with their ledgers, so the labelled path works out
of the box:

```bash
pdfplay samples
pdfplay add samples/bank_statement_borderless.pdf --class bank_statement \
        --ledger samples/bank_statement_borderless.ledger.json
pdfplay run <doc_id> --all && pdfplay score <doc_id>
```

Three variants are generated on purpose, because they separate parsers
differently: `ruled` (table with ruling lines), `borderless` (alignment only —
the common real case), and `scanned` (rasterized and slightly skewed, no text
layer at all, so every text-layer parser should score ~zero and OCR/vision
should not).

## The viewer

`pdfplay serve` opens a three-pane UI:

- **Left** — documents, parsers with availability and per-parser options.
- **Middle** — the rendered page with box overlays. Toggle layers, switch
  parsers, hover a box to see its text, click to select it in the text pane.
  Compare mode puts two parsers side by side on the same page.
- **Right** — extracted text / Markdown / tables / JSON, plus the scores.

## CLI

```
pdfplay list                       # parsers and availability
pdfplay samples [--out DIR]        # generate synthetic statements + ledgers
pdfplay add FILE [--class C] [--ledger L]
pdfplay docs                       # list documents in the workspace
pdfplay run DOC_ID (--all | -p parser)... [--pages 1,2] [--force] [--opt k=v]
pdfplay score DOC_ID [--class bank_statement] [--json]
pdfplay compare DOC_ID -p a -p b   # two-way diff between two parsers
pdfplay serve [--host H] [--port P]
```

`compare` reports similarity twice, and the gap between the two numbers is the
interesting part:

```
raw text similarity 0.7516   line similarity 1.0000   CER 0.4876
note: the same lines were recovered — the parsers differ in serialization order, not content
```

Raw similarity compares each parser's serialized text; line similarity compares
row-normalized lines. When raw is low and line is 1.0, the parsers extracted
identical content and disagree only about reading order — which matters for
prompt-stuffing an LLM but not for row-wise extraction.

The workspace lives in `./workspace` by default; override with
`PDFPLAY_WORKSPACE`.

## Adding a parser

Subclass `PdfParser`, return the normalized model, register it:

```python
from pdfplay.parsers.base import Option, PdfParser
from pdfplay.models import BBox, Block, PageResult, ParsedDocument
from pdfplay.registry import register

@register
class MyParser(PdfParser):
    id = "myparser"
    name = "My Parser"
    kind = "local"                 # or "remote"
    requires = ("mylib",)          # probed for availability
    env_vars = ()                  # e.g. ("MY_API_KEY",)
    options = (Option("dpi", "int", 200),)

    def parse(self, pdf_path, pages, options):
        ...
        return ParsedDocument(pages=[...])
```

Timing, error capture, caching, availability reporting, and the UI controls for
your options all come for free. For a remote vision model, subclass
`VisionParser` instead and implement only `call_model` — the prompt, schema,
rasterization, and box denormalization are shared.

## Adding a document class

The bank-statement scorer is a module with two entry points:
`analyze(result) -> report` (no labels needed) and
`score_against_ledger(result, truth)` (labelled). To add, say, invoices, write
`pdfplay/metrics/invoice.py` with the same two functions, register it in
`pdfplay/metrics/__init__.py::DOC_CLASSES`, and tag documents with
`pdfplay add FILE --class invoice`. The CLI, API, and scores pane pick it up
from there.

The pattern worth copying is the self-check: find the internal consistency the
document class already guarantees (a statement's running balance, an invoice's
line items summing to its total, a trial balance netting to zero) and score
against *that*. It costs nothing to label and it catches the failures that
matter.

## Known limits

- `docling`, `unstructured`, and `marker`-style adapters pull in heavy ML deps;
  they're behind extras and are exercised through the same interface, but the
  bundled test suite only covers the always-installed parsers.
- Vision-model boxes are approximate. Use them for region-level comparison, not
  for pixel-accurate cropping — that's what the text-layer parsers are for.
- The bank-statement row extractor is heuristic (date-prefixed lines). It's
  deliberately the *same* heuristic for every parser, so it measures parsers
  rather than itself, but it will under-count statements whose rows don't lead
  with a date.
