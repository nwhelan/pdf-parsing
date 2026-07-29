// Typed client for the pdfplay FastAPI backend. Types mirror the pydantic
// models in src/pdfplay/models.py and the response shapes in server/app.py.

export interface BBox {
  x0: number
  y0: number
  x1: number
  y1: number
}

export interface Block {
  id: string
  page: number
  layer: string
  kind: string
  text: string
  bbox: BBox | null
  order: number | null
  confidence: number | null
  meta: Record<string, unknown>
}

export interface TableCell {
  row: number
  col: number
  row_span: number
  col_span: number
  text: string
  bbox: BBox | null
  is_header: boolean
}

export interface TableData {
  id: string
  page: number
  bbox: BBox | null
  n_rows: number
  n_cols: number
  cells: TableCell[]
  meta: Record<string, unknown>
}

export interface PageResult {
  page_number: number
  width: number
  height: number
  rotation: number
  blocks: Block[]
  tables: TableData[]
  text: string
  meta: Record<string, unknown>
}

export interface Usage {
  input_tokens: number | null
  output_tokens: number | null
  cost_usd: number | null
  model: string | null
  requests: number
}

export interface ParseResult {
  parser_id: string
  parser_name: string
  parser_version: string
  doc_id: string
  doc_name: string
  status: string
  error: string | null
  duration_s: number
  per_page_s: Record<number, number>
  created_at: string
  options: Record<string, unknown>
  pages: PageResult[]
  markdown: string | null
  extraction: unknown
  usage: Usage
  warnings: string[]
  debug: Record<string, unknown>[]
}

export interface ParserOption {
  name: string
  type: string
  default: unknown
  label: string
  choices: unknown[] | null
  help: string
}

export interface ParserSpec {
  id: string
  name: string
  kind: string
  description: string
  homepage: string
  tags: string[]
  extra: string
  cost_hint: string
  options: ParserOption[]
  available: boolean
  unavailable_reason: string
  version: string
}

export interface Preset {
  preset_id: string
  name: string
  parser_id: string
  options: Record<string, unknown>
  created_at: string
  notes: string
}

export interface DocumentMeta {
  doc_id: string
  name: string
  pages: number
  size_bytes: number
  sha256: string
  added_at: string
  doc_class: string
  notes: string
}

export interface DocumentDetail extends DocumentMeta {
  geometry: { page: number; width: number; height: number; rotation: number }[]
  results: {
    key: string
    parser_id: string
    parser_name: string
    status: string
    error: string | null
    duration_s: number
    created_at: string
    options: Record<string, unknown>
    n_pages: number
  }[]
  has_ground_truth: boolean
}

export interface ExtractionScore {
  accuracy: number | null
  precision: number | null
  recall: number | null
  f1: number | null
  n_fields: number
  n_correct: number
  fields: {
    path: string
    status: "correct" | "wrong" | "missing" | "extra"
    expected: unknown
    actual: unknown
    similarity?: number
  }[]
}

export interface ScoreRow {
  key: string
  parser_id: string
  status: string
  error: string | null
  seconds_per_page: number | null
  n_chars: number
  n_lines: number
  page_coverage: number | null
  reading_order_score: number | null
  cost_usd: number | null
  bank_statement?: {
    n_transactions: number
    reconciliation_rate: number
    totals_match: boolean | null
    amount_column_consistency: number | null
    amount_columns_detected: number
    balance_column_consistency: number | null
    breaks: { page: number; expected_delta: string; amount: string; raw: string }[]
  }
  ledger_score?: {
    precision: number | null
    recall: number | null
    f1: number | null
    description_similarity: number | null
    balance_accuracy: number | null
  }
  extraction_score?: ExtractionScore
}

export interface ScoreResponse {
  doc_class: string
  known_classes: string[]
  has_golden_extraction: boolean
  rows: ScoreRow[]
  similarity: Record<string, Record<string, number>>
}

export interface DiffResponse {
  text_similarity: number
  line_similarity: number
  cer: number
  same_content_different_order: boolean
  hunks: { left: string[]; right: string[] }[]
}

export const LAYER_ORDER = ["word", "line", "block", "region", "table", "cell"]

export const LAYER_COLOR: Record<string, string> = {
  word: "#3b82f6",
  line: "#10b981",
  block: "#f59e0b",
  region: "#a855f7",
  table: "#ef4444",
  cell: "#06b6d4",
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (typeof body?.detail === "string") detail = body.detail
    } catch {
      /* not JSON */
    }
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<T>
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
})

export const api = {
  parsers: () => request<ParserSpec[]>("/api/parsers"),

  presets: () => request<Preset[]>("/api/presets"),
  savePreset: (name: string, parser_id: string, options: Record<string, unknown>) =>
    request<Preset>("/api/presets", json({ name, parser_id, options })),
  deletePreset: (presetId: string) =>
    request<{ ok: boolean }>(`/api/presets/${presetId}`, { method: "DELETE" }),

  documents: () => request<DocumentMeta[]>("/api/documents"),
  document: (docId: string) => request<DocumentDetail>(`/api/documents/${docId}`),
  upload: (file: File) => {
    const form = new FormData()
    form.append("file", file)
    return request<DocumentMeta>("/api/documents", { method: "POST", body: form })
  },

  result: (docId: string, key: string) =>
    request<ParseResult>(`/api/documents/${docId}/results/${key}`),
  parse: (docId: string, parserId: string, payload: { options: Record<string, unknown>; pages?: number[]; force?: boolean }) =>
    request<{ key: string; cached: boolean; result: ParseResult }>(
      `/api/documents/${docId}/parse/${parserId}`,
      json(payload)
    ),

  score: (docId: string, keys: string[]) =>
    request<ScoreResponse>(`/api/documents/${docId}/score`, json({ keys })),
  diff: (docId: string, left: string, right: string) =>
    request<DiffResponse>(`/api/documents/${docId}/diff`, json({ left, right })),
}
