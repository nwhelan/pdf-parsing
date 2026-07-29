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
| `pymupdf-layout` | local (extra) | `pymupdf4llm.to_markdown` with the pymupdf-layout ONNX model: classified regions, real Markdown tables, and an OCR fallback. Switch `engine=classic` for the same call without the model. |
| `pdfplumber` | local | pdfminer + word clustering. Its `text` table strategy handles **borderless** tables. |
| `pypdfium2` | local | Chrome's PDF engine. Text rects only — a good "what's really in the text layer" baseline. |
| `pdfminer` | local | Tunable `LAParams`; shows how sensitive paragraph grouping is on a document class. |
| `docling` | local (extra) | Layout model + TableFormer + optional OCR. Labelled regions, Markdown export. |
| `unstructured` | local (extra) | Element classification, `hi_res` layout model. |
| `tesseract` | local (extra) | Rasterize + OCR, ignoring the text layer. The scanned-document control. |
| `claude` | remote (extra) | Page image → structured JSON transcription, via Anthropic structured outputs. |
| `openai` | remote (extra) | Same contract via Chat Completions `json_schema`. |
| `gemini` | remote (extra) | Same contract, using Gemini's native `[ymin,xmin,ymax,xmax]` box convention. |
| `mistral-ocr-3` | remote (extra) | Document-level OCR API: Markdown per page including tables. Endpoint is pluggable — Mistral, Azure AI Foundry, Vertex, or a gateway. |
| `mistral-ocr-4` | remote (extra) | The same adapter on the newer model, registered separately so the two generations can run side by side. |
| `openai-compatible` | remote (extra) | The Chat Completions adapter with no vendor assumptions. Point `base_url` at Azure OpenAI, a LiteLLM proxy, vLLM, Ollama, OpenRouter, Together or Mistral. |

Install the optional ones as needed:

```bash
uv pip install -e '.[openai]'       # also: layout, docling, unstructured, ocr, anthropic, gemini, mistral, vision
```

Remote parsers read their credentials from the environment
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`) and
report themselves as unavailable — with the reason — when a key is missing.
Nothing is ever sent anywhere unless you explicitly run a remote parser.

## Bring your own endpoint

Every remote model here is served from more than one place, and which place you
use is a property of your account, not of the model. So endpoints are options
rather than constants.

Everything that speaks the Chat Completions protocol goes through the **OpenAI
SDK** — `OpenAI` for anything serving it, `AzureOpenAI` when an api-version says
Azure. No routing layer in between, which means one dependency, one client, and
errors that come back from the provider rather than from a translation of it.

```bash
pdfplay run <doc_id> -p openai-compatible \
  --opt base_url=http://localhost:4000/v1 --opt model=my-gateway-model
```

That reaches Azure OpenAI, a LiteLLM proxy, vLLM, Ollama, OpenRouter, Together,
Fireworks or Mistral's chat models. Set `api_version` and it switches to the
Azure OpenAI client, reading `base_url` as the resource endpoint and `model` as
a deployment name; without it the plain client is used, and an Azure resource
URL then needs to end in `/openai/v1`.

The token limit is sent as `max_completion_tokens` for GPT-5 and the o-series
and as `max_tokens` for everything else. An Azure deployment can be named
anything, so if the API rejects one spelling the call is retried once with the
other rather than making it your problem — `token_param` forces either. `api_key_env` names the variable holding the key; a local
server that wants no key needs nothing. `response_format` drops from
`json_schema` to `json_object` to `text` for servers that only partly implement
structured output.

The stock `openai` parser takes the same options, so pointing *it* somewhere
else works too — `openai-compatible` exists so both can sit in one comparison.

### Named models from a config.yaml

If you keep a LiteLLM-style config of models and endpoints, the `model` option
can name an entry in it and the endpoint comes along. The file format is
LiteLLM's because that is what people already have on disk; reading it needs
nothing but PyYAML.

```yaml
model_list:
  - model_name: statement-vision
    litellm_params:
      model: azure/gpt-4.1-deployment
      api_base: https://my-resource.openai.azure.com
      api_key: os.environ/AZURE_OPENAI_API_KEY
      api_version: "2026-01-01"
```

```bash
pdfplay run <doc_id> -p openai-compatible --opt model=statement-vision
```

The config is found via the `config_path` option, then `$PDFPLAY_MODEL_CONFIG` /
`$LITELLM_CONFIG_PATH`, then `./config.yaml`. A `provider/` prefix on the model
is stripped — the URL already decides who serves it — and `api_version` selects
the Azure client. Configured names appear as suggestions on the `model` field in
the viewer, and the field stays free text, so any model id still works. Anything
set explicitly (`base_url`, `api_key_env`) overrides the file, and
`os.environ/NAME` keys are resolved per call, never stored in a result, a preset
or a cache key.

Two things sit outside this. **Mistral OCR** is a document API, not a chat one —
there is no `/v1/ocr` in the OpenAI protocol and no SDK client that speaks it —
so those adapters POST directly and keep their own endpoint options. **Claude
and Gemini** keep their native SDKs, where their own features live; reach them
over the OpenAI protocol via `openai-compatible` and a gateway if you prefer.

## Comparing models, not just libraries

A library comparison is settled by the library. A *model* comparison is only
meaningful if every model was asked the same question in the same shape, so
three things are options rather than constants.

**Instructions.** Every vision parser takes `instructions`, applied to each
page. `instructions_mode=append` (the default) adds them under the standard
transcription prompt; `replace` hands the model your text alone.

```bash
pdfplay run <doc_id> -p claude -p openai -p gemini \
  --opt instructions='Treat parentheses as negative amounts. Never reformat dates.'
```

**An extraction schema.** Set `extraction_schema` to a JSON Schema and the
model returns an `extraction` object matching it *alongside* the transcription —
one call, so you can score the structured answer and the page text together.
The same schema across models is the whole point:

```bash
SCHEMA='{"type":"object","properties":{
  "account_number":{"type":"string"},
  "closing_balance":{"type":"number"}}}'
pdfplay run <doc_id> -p claude --opt extraction_schema="$SCHEMA"
pdfplay run <doc_id> -p openai-compatible --opt model=statement-vision --opt extraction_schema="$SCHEMA"
```

Mistral OCR has its own native version of this, so it can answer the same
question through its own API rather than through a prompt — see below. Whichever
route produced it, the result lands in the same `extraction` field and the same
**Extraction** tab in the viewer.

**Presets.** An endpoint, a deployment name, a prompt and a schema are a lot to
retype, and a comparison you can't re-run tomorrow isn't one. Save the lot under
a name:

```bash
pdfplay presets --save "Foundry OCR" -p mistral-ocr-3 \
        --opt endpoint=azure --opt model=mistral-document-ai-2512
pdfplay presets                                   # list them
pdfplay run <doc_id> --preset "Foundry OCR"       # implies its parser
pdfplay presets --delete mistral-ocr-3__foundry-ocr
```

`--preset` can be repeated and mixed with `-p`, and an explicit `--opt`
overrides the stored value for that run. In the viewer, the same presets live at
the top of the options panel: type a name, hit Save, click to re-apply. They're
stored in `workspace/presets.json`, so they travel with the workspace.

### Mistral OCR

Mistral OCR is document-level rather than page-image-level: you post the whole
PDF and get Markdown back per page, plus boxes for any *images* it found. It
returns no geometry for text, so the overlay shows figures only — the comparison
value is in the Markdown, the recovered tables, and whether the ledger
reconciles. Markdown tables are parsed back into the normalized `Table` model
(the same code path `pymupdf-layout` uses), so its tables are comparable with
parsers that expose real cell geometry.

The call is a direct `httpx` POST, since this endpoint is not part of the
OpenAI protocol and no SDK client speaks it.

| Option | Effect |
|---|---|
| `endpoint` | `mistral` (default), `azure`, or `custom`. Picks the URL and the auth scheme. |
| `base_url` | The OCR endpoint. An Azure resource root is completed for you. |
| `api_key_env` | Env var holding the key. Blank tries `MISTRAL_API_KEY`, `AZURE_AI_API_KEY`, `AZURE_MISTRAL_API_KEY`, `MISTRAL_OCR_API_KEY`. |
| `auth_header` | `auto` sends Bearer to Mistral and `api-key` to Azure. Override for gateways. |
| `api_version` | Azure only: value for the `?api-version=` query parameter. |
| `model` | Model id, or an Azure *deployment* name. |

Foundry serves OCR under two conventions depending on how the model was
deployed, and the path follows the hostname: `*.models.ai.azure.com` (serverless)
gets `/v1/ocr`, an AI Services resource gets `/providers/mistral/azure/ocr`. A
bare hostname is completed either way — posting a document to a host root is
never what was meant. A URL that already has a path is left alone, since it may
point at a gateway. Note that `*.openai.azure.com` is Azure **OpenAI**, which
does not serve Mistral OCR at all.

Because options are part of the cache key, `mistral-ocr-3` can target an Azure
Foundry deployment while `mistral-ocr-4` talks to `api.mistral.ai` in the same
workspace, and their results are cached separately.

**Model ids.** Mistral names OCR releases by date, not by generation, so the
parser ids here are labels and the model strings are what actually gets sent:

| Where | Ids |
|---|---|
| Mistral API | `mistral-ocr-latest`, `mistral-ocr-2505-completion`, `mistral-ocr-2512` (the `mistral-ocr-3` default), `mistral-ocr-4-0` (the `mistral-ocr-4` default) |
| Azure AI Foundry | `mistral-document-ai-2505`, `mistral-document-ai-2512` |
| Vertex AI | `mistral-ocr-2505` |

Set `model` to whatever your account exposes. A test pins the two defaults to
that list, so a stale id fails the suite rather than the first API call.

**Data extraction.** Mistral OCR takes a JSON Schema and returns structured
fields next to the Markdown, which is a different mechanism from prompting a
vision model for JSON and worth comparing against one. Three options carry it:

| Option | Sent as | Effect |
|---|---|---|
| `document_annotation_schema` | `document_annotation_format` | Fields extracted from the document as a whole. |
| `document_annotation_prompt` | `document_annotation_prompt` | Instructions to go with that schema. |
| `bbox_annotation_schema` | `bbox_annotation_format` | Applied to each image region — captions, chart summaries. |

Paste a bare schema; the `json_schema` envelope the API wants is added for you,
and an already-wrapped one is passed through untouched. The answer comes back
JSON-decoded in the result's `extraction` field — the same field the vision
parsers' `extraction_schema` fills, so the two approaches line up in the
**Extraction** tab.

```bash
pdfplay run <doc_id> -p mistral-ocr-3 \
  --opt document_annotation_schema='{"type":"object","properties":{"closing_balance":{"type":"number"}}}' \
  --opt document_annotation_prompt='Read the balance from the summary box, not the ledger.'
```

**Azure AI Foundry.** The bare resource root is enough — the OCR path is
completed for you:

```bash
export AZURE_AI_API_BASE=https://my-resource.services.ai.azure.com
export AZURE_AI_API_KEY=...
pdfplay run <doc_id> -p mistral-ocr-3 \
  --opt endpoint=azure --opt model=mistral-document-ai-2512
# → POST https://my-resource.services.ai.azure.com/providers/mistral/azure/ocr
```

`AZURE_MISTRAL_ENDPOINT` / `AZURE_MISTRAL_API_KEY` work too; the `AZURE_AI_*`
names are the conventional Foundry ones, so a workspace already configured for
it needs nothing extra. Give `base_url` the full URL if your deployment doesn't follow
that path. In the viewer all of these are ordinary fields in the parser's
options panel.

> The Mistral hosts are unreachable from the environment this was written in
> (egress policy returns 403 for `api.mistral.ai` and `docs.mistral.ai`), so
> none of it was exercised against a live endpoint. The request shape, the model
> ids and the annotation parameters were cross-checked against an independent
> implementation of the same API, which is real corroboration but not the same
> as a successful call.

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

Row extraction is format-agnostic on purpose: a parser that reconstructs the
ledger as a Markdown table (`|03/02/2025|CHECK 1042|31.61||6,492.77|`) is
producing *better* output than a flat line, so pipes are unwrapped before the
row heuristics run rather than counting against it.

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

### Golden sets for extraction

The ledger scorer is specific to statements. For structured extraction there's a
document-class-agnostic one: store the right answer for a document, and every
parser's `extraction` is scored against it field by field.

```bash
pdfplay golden <doc_id> --set golden.json     # the known-correct answer
pdfplay run <doc_id> -p claude -p openai --opt extraction_schema="$SCHEMA"
pdfplay score <doc_id>                        # adds an extraction table
pdfplay score <doc_id> --fields               # every field, expected vs got
```

```
parser        fields  right    acc      P      R     F1
claude             6      6   1.00   1.00   1.00   1.00
openai             6      4   0.67   0.80   0.67   0.73

openai
  ✓ account_number                   want='0042-118-9' got='0042-118-9'
  ✓ closing_balance                  want=5078.59 got='$5,078.59'
  ✗ holder.name                      want='A. Nwosu' got='A Nwoso'
```

The per-field verdict is the point — a single accuracy number tells you
something went wrong, not what. Four statuses: `correct`, `wrong` (answered,
but not right — carrying a similarity so near-misses are visible as such),
`missing` (didn't answer) and `extra` (answered something the golden doesn't
mention, which costs precision but not recall). Not answering and answering
wrongly are deliberately different: a parser that returns nulls everywhere gets
zero recall rather than perfect precision.

Comparison is by value, not by formatting, because parsers disagree about
formatting and chasing that is wasted time: `$5,078.59`, `5,078.59` and
`5078.59` are the same number, `(1,708.14)` is negative, `03/31/2025` and
`2025-03-31` are the same date, and text is compared ignoring case, spacing and
punctuation. Nested objects and lists are addressed by path
(`fees[0].amount`), so a golden set describes whatever shape your schema has.

Vision parsers answer per page and Mistral answers per document; the per-page
form is merged before scoring — first non-null answer per field wins, and lists
accumulate — so both are scored against the same golden set.

The golden lives beside any ledger in the document's `ground_truth.json`, and
the viewer shows the same comparison as a table in the **Extraction** tab.

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

On the bundled samples that separation looks like this — every text-layer
parser reads the clean statement perfectly, and the scan is where the field
splits:

```
scanned          txns  recon   F1     s/pg
pymupdf             0   0.00   0.00   0.07     text layer is empty
pdfplumber          0   0.00   0.00   0.10
pdfminer            0   0.00   0.00   0.10
pypdfium2           0   0.00   0.00   0.09
tesseract          34   1.00   1.00   1.52     OCR
pymupdf-layout     34   1.00   1.00   2.75     layout model + OCR fallback
```

## The viewer

`pdfplay serve` opens a React app built with [shadcn/ui](https://ui.shadcn.com)
components on Tailwind v4:

- **Left** — a resizable, collapsible `Sidebar` listing parsers grouped
  local/remote, each
  with a checkbox to run it and a tooltip explaining what it does (or why it
  isn't available). Selecting one opens its options below, rendered from the
  adapter's own `Option` declarations — `Switch` for booleans, `Select` for
  choices, `Input` for numbers. Nothing about the UI is parser-specific.
- **Middle** — the rendered page inside `Resizable` panels, with each parser's
  boxes drawn on top. A `ToggleGroup` switches granularity layers
  (word/line/block/region/table/cell), hovering a box shows its text in a
  `Tooltip`, and clicking selects it in the text pane. Compare mode splits the
  pane so two parsers sit side by side on the same page; fit mode scales each
  pane independently so both stay legible.
- **Right** — `Tabs` over scores, text, Markdown, tables, diff, and raw JSON.
  Scores render as `Card` sections: generic signals in a `Table`, per-parser
  reconciliation as a `Progress` bar, ground-truth P/R/F1, and the text
  agreement matrix. Run results and failures arrive as toasts.

The sidebar has no fixed width: drag its right edge to resize, click it to
collapse, double-click to reset, or focus it and use the arrow keys (shift for
bigger steps). The width persists across reloads. Prompts and JSON schemas live
in that rail, so it needs to be able to grow.

Dark and light themes both ship; the toggle is in the header.

### When a parser fails

A failure produces a toast that goes away and an entry that doesn't. The toast
names the parser and disappears after a few seconds; the entry lands in a
failures log, reachable from the ⚠ count in the header or the *Details* button
on the scores pane. Failed results are stored like any other, so the log is
seeded from them on reload — a failure from yesterday is still inspectable.

Each entry carries **the request that preceded it**, which is usually the only
way to read the error. `wire` is exactly what goes to the API; everything about
how it was decided lives under `context`, so the two can't be confused:

```json
{
  "event": "chat.completions.create",
  "wire": {
    "model": "gpt-5.2",
    "max_completion_tokens": 16000,
    "messages": [{ "role": "user", "content": ["image_url", "text"] }]
  },
  "context": {
    "client": "OpenAI",
    "base_url": "https://my-resource.openai.azure.com",
    "api_version": null,
    "model_source": "option",
    "token_param": "max_completion_tokens",
    "image_bytes": 652538
  },
  "hints": [
    "my-resource.openai.azure.com is an Azure OpenAI resource, and without api_version the plain OpenAI client is used — that needs base_url to end in /openai/v1."
  ]
}
```

**`hints`** name configurations that will fail before the API refuses them,
because a 404 rarely says which of several conventions was expected: an Azure
OpenAI resource root used without an api-version, Mistral OCR pointed at an
Azure *OpenAI* host (which doesn't serve it at all), a URL with no path, or a
`mistral-ocr-*` id where Foundry wants a deployment name.

`DeploymentNotFound` is a different problem depending on whether `model_source`
says `option` or `config.yaml`, and a 400 asking for a
`document_annotation_prompt` is only readable next to the payload that omitted
it. Adapters record before they call, so the request survives the failure.

Turn on a parser's **`debug`** option and the log also keeps the full prompt, the
image data URL and the raw response body — the whole request as sent, for when
the shape itself is in question. Off by default because the bodies are large.

Credentials never reach the log: anything whose key looks like a secret is
masked, and long values (a base64 PDF, an image) are truncated to their shape.
The log is part of the stored result, so it's in the JSON tab too.

If the viewer says **"Could not load parsers: failed to fetch"**, the page
reached the static bundle but not the API. It retries five times over about six
seconds first, then leaves a toast with a Retry button, so a server that was
still binding its port recovers on its own. If it doesn't, the terminal running
`pdfplay serve` has the reason — it logs every request at `info` by default;
`--log-level debug` if you need more. A parser whose availability probe raises
is reported as unavailable with the exception on its tooltip, rather than
taking the whole list down with it.

### Working on the front-end

The build output is committed under `src/pdfplay/server/static`, so
`pip install` + `pdfplay serve` works with no node step. To change the UI:

```bash
cd web
npm ci
npm run dev        # vite on :5173, proxying /api to pdfplay serve on :8000
npm run build      # writes back into src/pdfplay/server/static
```

The components live in `web/src/components/ui` and are yours to edit — that is
shadcn's model, not a limitation. Note that this session could not reach
`ui.shadcn.com` (egress policy), so the components were written into the
project directly rather than pulled with `npx shadcn add`; `components.json` is
present so the CLI works normally wherever the registry is reachable.

## CLI

```
pdfplay list                       # parsers and availability
pdfplay samples [--out DIR]        # generate synthetic statements + ledgers
pdfplay add FILE [--class C] [--ledger L]
pdfplay docs                       # list documents in the workspace
pdfplay run DOC_ID (--all | -p parser | --preset name)... [--pages 1,2] [--force] [--opt k=v]
pdfplay presets [-p parser] [--save NAME --opt k=v] [--delete PRESET_ID]
pdfplay golden DOC_ID [--set FILE.json]
pdfplay score DOC_ID [--class bank_statement] [--fields] [--json]
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
`PDFPLAY_WORKSPACE` or `pdfplay --workspace DIR ...`. Everything it writes —
results, metadata, presets — is UTF-8 regardless of platform, so a parser that
returns an em dash or a currency symbol doesn't fail on save under a Windows
code page.

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

Before writing one, check whether the model already speaks the OpenAI protocol —
`openai-compatible` with a `base_url` and a `model` reaches it without any new
code. A new adapter is worth it when the provider has something the shared
contract can't express.

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
