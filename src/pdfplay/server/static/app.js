/* pdfplay viewer: page render + box overlays + comparison. */

const LAYER_COLORS = {
  word: "#6aa9ff",
  line: "#4ec9a0",
  block: "#e0a33e",
  region: "#c678dd",
  table: "#e06c75",
  cell: "#56b6c2",
};

const state = {
  parsers: [],
  doc: null,
  page: 1,
  zoom: 1.5,
  selected: new Set(),          // parser ids ticked for running
  optionsFor: null,             // parser id whose options are shown
  optionValues: {},             // parserId -> {opt: value}
  results: {},                  // key -> full result
  panes: { a: { key: null, layers: new Set(["line"]) }, b: { key: null, layers: new Set(["line"]) } },
  scores: null,
  tab: "scores",
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function status(msg, isError) {
  const el = $("#status");
  el.textContent = msg || "";
  el.className = "status" + (isError ? " err" : "");
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.headers.get("content-type")?.includes("json") ? res.json() : res.text();
}

/* ---------------- parsers ---------------- */

async function loadParsers() {
  state.parsers = await api("/api/parsers");
  const box = $("#parser-list");
  box.innerHTML = "";
  for (const p of state.parsers) {
    const row = document.createElement("div");
    row.className = "parser" + (p.available ? "" : " unavailable");
    row.innerHTML = `
      <input type="checkbox" ${p.available ? "" : "disabled"} data-id="${p.id}" />
      <span class="dot ${p.available ? p.kind : "off"}"></span>
      <div>
        <div class="name">${esc(p.name)}</div>
        <div class="meta">${esc(p.available ? (p.version || p.kind) : p.unavailable_reason)}</div>
      </div>`;
    row.querySelector("input").addEventListener("change", (e) => {
      e.target.checked ? state.selected.add(p.id) : state.selected.delete(p.id);
    });
    row.addEventListener("click", (e) => {
      if (e.target.tagName === "INPUT") return;
      state.optionsFor = p.id;
      $$(".parser").forEach((n) => n.classList.remove("selected-for-options"));
      row.classList.add("selected-for-options");
      renderOptions(p);
    });
    box.appendChild(row);
  }
}

function renderOptions(parser) {
  const panel = $("#options-panel");
  if (!parser.options.length) {
    panel.innerHTML = `<div class="muted">${esc(parser.name)} has no options.<br><br>${esc(parser.description)}</div>`;
    return;
  }
  const values = (state.optionValues[parser.id] ||= {});
  panel.innerHTML = `<div class="muted" style="margin-bottom:8px">${esc(parser.description)}</div>`;
  for (const opt of parser.options) {
    const current = values[opt.name] ?? opt.default;
    const row = document.createElement("div");
    row.className = "opt-row";
    let input;
    if (opt.type === "bool") {
      input = `<input type="checkbox" ${current ? "checked" : ""} />`;
    } else if (opt.type === "choice") {
      input = `<select>${opt.choices.map((c) => `<option ${String(c) === String(current) ? "selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
    } else if (opt.type === "int" || opt.type === "float") {
      input = `<input type="number" step="${opt.type === "int" ? 1 : "any"}" value="${esc(current)}" />`;
    } else {
      input = `<input type="text" value="${esc(current)}" />`;
    }
    row.innerHTML = `<label title="${esc(opt.help)}">${esc(opt.label)}</label>${input}`;
    const el = row.querySelector("input, select");
    el.addEventListener("change", () => {
      if (opt.type === "bool") values[opt.name] = el.checked;
      else if (opt.type === "int") values[opt.name] = parseInt(el.value, 10);
      else if (opt.type === "float") values[opt.name] = parseFloat(el.value);
      else values[opt.name] = el.value;
    });
    panel.appendChild(row);
    if (opt.help) {
      const help = document.createElement("div");
      help.className = "opt-help";
      help.textContent = opt.help;
      panel.appendChild(help);
    }
  }
}

/* ---------------- documents ---------------- */

async function loadDocuments(selectId) {
  const docs = await api("/api/documents");
  const sel = $("#doc-select");
  sel.innerHTML = docs.map((d) => `<option value="${d.doc_id}">${esc(d.name)} · ${d.pages}p${d.doc_class ? " · " + esc(d.doc_class) : ""}</option>`).join("");
  if (!docs.length) {
    sel.innerHTML = `<option value="">no documents — add a PDF</option>`;
    return;
  }
  sel.value = selectId || docs[0].doc_id;
  await selectDocument(sel.value);
}

async function selectDocument(docId) {
  state.doc = await api(`/api/documents/${docId}`);
  state.page = 1;
  state.results = {};
  for (const r of state.doc.results) {
    if (r.status === "ok" || r.status === "error") {
      state.results[r.key] = await api(`/api/documents/${docId}/results/${r.key}`);
    }
  }
  const keys = Object.keys(state.results);
  state.panes.a.key = keys[0] || null;
  state.panes.b.key = keys[1] || keys[0] || null;
  refreshResultSelects();
  renderPage();
  await refreshScores();
}

/* ---------------- running ---------------- */

async function runSelected() {
  if (!state.doc) return;
  const ids = Array.from(state.selected);
  if (!ids.length) return status("tick a parser first", true);
  const force = $("#force-rerun").checked;
  for (const id of ids) {
    status(`running ${id}…`);
    try {
      const payload = { options: state.optionValues[id] || {}, force };
      const out = await api(`/api/documents/${state.doc.doc_id}/parse/${id}`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      state.results[out.key] = out.result;
      if (out.result.status === "error") status(`${id}: ${out.result.error}`, true);
    } catch (err) {
      status(`${id}: ${err.message}`, true);
    }
  }
  const keys = Object.keys(state.results);
  if (!state.panes.a.key) state.panes.a.key = keys[0];
  if (!state.panes.b.key) state.panes.b.key = keys[1] || keys[0];
  refreshResultSelects();
  renderPage();
  await refreshScores();
  status("");
}

/* ---------------- overlay rendering ---------------- */

function refreshResultSelects() {
  const options = Object.entries(state.results)
    .map(([key, r]) => `<option value="${key}">${esc(r.parser_name || r.parser_id)}${r.status === "error" ? " ⚠" : ""}</option>`)
    .join("");
  for (const pane of ["a", "b"]) {
    const sel = document.querySelector(`.result-select[data-pane="${pane}"]`);
    sel.innerHTML = options || `<option value="">no results yet</option>`;
    if (state.panes[pane].key) sel.value = state.panes[pane].key;
  }
  renderLayerChips();
}

function renderLayerChips() {
  for (const pane of ["a", "b"]) {
    const host = document.querySelector(`.layers[data-pane="${pane}"]`);
    const result = state.results[state.panes[pane].key];
    host.innerHTML = "";
    if (!result) continue;
    const layers = new Set();
    for (const p of result.pages) for (const b of p.blocks) layers.add(b.layer);
    const active = state.panes[pane].layers;
    // Default to the finest layer that exists, once.
    if (![...active].some((l) => layers.has(l))) {
      active.clear();
      for (const preferred of ["line", "region", "block", "word", "table"]) {
        if (layers.has(preferred)) { active.add(preferred); break; }
      }
    }
    for (const layer of ["word", "line", "block", "region", "table", "cell"]) {
      if (!layers.has(layer)) continue;
      const chip = document.createElement("span");
      const on = active.has(layer);
      chip.className = "layer-chip" + (on ? " on" : "");
      chip.textContent = layer;
      chip.style.borderColor = LAYER_COLORS[layer];
      if (on) chip.style.background = LAYER_COLORS[layer];
      chip.addEventListener("click", () => {
        on ? active.delete(layer) : active.add(layer);
        renderLayerChips();
        renderPage();
      });
      host.appendChild(chip);
    }
  }
}

function renderPage() {
  if (!state.doc) return;
  $("#page-label").textContent = `${state.page} / ${state.doc.pages}`;
  for (const pane of ["a", "b"]) drawPane(pane);
}

function drawPane(pane) {
  const host = document.querySelector(`.canvas[data-pane="${pane}"]`);
  const geo = (state.doc.geometry || []).find((g) => g.page === state.page);
  if (!geo) { host.innerHTML = ""; return; }

  const width = geo.width * state.zoom;
  host.style.width = `${width}px`;
  host.innerHTML = `<img src="/api/documents/${state.doc.doc_id}/pages/${state.page}/image?scale=2" alt="page ${state.page}" />`;

  const result = state.results[state.panes[pane].key];
  if (!result || result.status !== "ok") return;
  const page = result.pages.find((p) => p.page_number === state.page);
  if (!page) return;

  // Boxes are stored in PDF points; the render is `zoom` px per point.
  const scale = state.zoom;
  const active = state.panes[pane].layers;
  const showLabels = $("#show-labels").checked;

  for (const block of page.blocks) {
    if (!block.bbox || !active.has(block.layer)) continue;
    const el = document.createElement("div");
    el.className = "box";
    el.style.left = `${block.bbox.x0 * scale}px`;
    el.style.top = `${block.bbox.y0 * scale}px`;
    el.style.width = `${Math.max(1, (block.bbox.x1 - block.bbox.x0) * scale)}px`;
    el.style.height = `${Math.max(1, (block.bbox.y1 - block.bbox.y0) * scale)}px`;
    el.style.borderColor = LAYER_COLORS[block.layer] || "#888";
    el.dataset.text = block.text || "";
    el.addEventListener("mousemove", (e) => showTooltip(e, block));
    el.addEventListener("mouseleave", hideTooltip);
    el.addEventListener("click", () => {
      $$(".box").forEach((b) => b.classList.remove("selected"));
      el.classList.add("selected");
      showTab("text");
      highlightInText(block);
    });
    if (showLabels && block.kind && block.kind !== "text") {
      const label = document.createElement("span");
      label.className = "box-label";
      label.textContent = block.kind;
      label.style.background = LAYER_COLORS[block.layer] || "#888";
      el.appendChild(label);
    }
    host.appendChild(el);
  }
}

function showTooltip(event, block) {
  const tip = $("#tooltip");
  const conf = block.confidence != null ? ` · conf ${(block.confidence * 100).toFixed(0)}%` : "";
  tip.textContent = `[${block.layer}/${block.kind}${conf}]\n${block.text || "(no text)"}`.slice(0, 800);
  tip.style.left = `${Math.min(event.clientX + 14, window.innerWidth - 480)}px`;
  tip.style.top = `${event.clientY + 14}px`;
  tip.classList.remove("hidden");
}
const hideTooltip = () => $("#tooltip").classList.add("hidden");

/* ---------------- inspector ---------------- */

function showTab(name) {
  state.tab = name;
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab-body").forEach((b) => b.classList.add("hidden"));
  $(`#tab-${name}`).classList.remove("hidden");
  renderTab();
}

function renderTab() {
  const result = state.results[state.panes.a.key];
  if (state.tab === "scores") return renderScores();
  if (state.tab === "diff") return renderDiff();
  if (!result) { $(`#tab-${state.tab}`).innerHTML = `<div class="muted">Run a parser first.</div>`; return; }

  if (state.tab === "text") {
    const page = result.pages.find((p) => p.page_number === state.page);
    $("#tab-text").innerHTML = `<pre id="text-pre">${esc(page ? page.text : "")}</pre>`;
  } else if (state.tab === "markdown") {
    $("#tab-markdown").innerHTML = result.markdown
      ? `<pre>${esc(result.markdown)}</pre>`
      : `<div class="muted">${esc(result.parser_name)} does not emit Markdown.</div>`;
  } else if (state.tab === "tables") {
    const page = result.pages.find((p) => p.page_number === state.page);
    const tables = page ? page.tables : [];
    $("#tab-tables").innerHTML = tables.length
      ? tables.map((t, i) => `<h2>table ${i + 1} — ${t.n_rows}×${t.n_cols}</h2>${tableHtml(t)}`).join("")
      : `<div class="muted">No tables found on this page.</div>`;
  } else if (state.tab === "json") {
    const page = result.pages.find((p) => p.page_number === state.page);
    $("#tab-json").innerHTML = `<pre>${esc(JSON.stringify(page, null, 1))}</pre>`;
  }
}

function tableHtml(table) {
  const grid = Array.from({ length: table.n_rows }, () => Array(table.n_cols).fill(""));
  for (const c of table.cells) if (grid[c.row]) grid[c.row][c.col] = c.text;
  return `<table class="scores">${grid.map((row, i) =>
    `<tr>${row.map((cell) => `<${i === 0 ? "th" : "td"}>${esc(cell)}</${i === 0 ? "th" : "td"}>`).join("")}</tr>`).join("")}</table>`;
}

function highlightInText(block) {
  const pre = $("#text-pre");
  if (!pre || !block.text) return;
  const idx = pre.textContent.indexOf(block.text.trim().slice(0, 40));
  if (idx < 0) return;
  const range = document.createRange();
  range.setStart(pre.firstChild, idx);
  range.setEnd(pre.firstChild, Math.min(idx + block.text.length, pre.textContent.length));
  const sel = window.getSelection();
  sel.removeAllRanges();
  sel.addRange(range);
  pre.scrollTop = Math.max(0, (idx / pre.textContent.length) * pre.scrollHeight - 100);
}

async function refreshScores() {
  if (!state.doc || !Object.keys(state.results).length) { state.scores = null; renderScores(); return; }
  state.scores = await api(`/api/documents/${state.doc.doc_id}/score`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ keys: Object.keys(state.results) }),
  });
  if (state.tab === "scores") renderScores();
}

function cls(value, good, mid) {
  if (value == null) return "";
  if (value >= good) return "good";
  if (value >= mid) return "mid";
  return "bad";
}

function renderScores() {
  const host = $("#tab-scores");
  if (!state.scores) { host.innerHTML = `<div class="muted">Run some parsers to see scores.</div>`; return; }
  const rows = state.scores.rows;
  const isBank = state.scores.doc_class === "bank_statement";

  let html = `<h2>Generic signals</h2><table class="scores">
    <tr><th>parser</th><th>s/page</th><th>chars</th><th>lines</th><th>cover</th><th>order</th><th>dup</th><th>$</th></tr>`;
  for (const r of rows) {
    if (r.status !== "ok") { html += `<tr><td>${esc(r.parser_id)}</td><td colspan="7" class="bad">${esc(r.error || "error")}</td></tr>`; continue; }
    html += `<tr>
      <td>${esc(r.parser_id)}</td>
      <td>${r.seconds_per_page.toFixed(2)}</td>
      <td>${r.n_chars}</td>
      <td>${r.n_lines}</td>
      <td>${r.page_coverage}</td>
      <td class="${cls(r.reading_order_score, .95, .8)}">${r.reading_order_score}</td>
      <td class="${r.duplicate_line_ratio > .1 ? "bad" : ""}">${r.duplicate_line_ratio}</td>
      <td>${r.cost_usd != null ? "$" + r.cost_usd.toFixed(4) : "–"}</td>
    </tr>`;
  }
  html += `</table>`;

  if (isBank) {
    html += `<h2>Bank statement</h2><table class="scores">
      <tr><th>parser</th><th>txns</th><th>recon</th><th>amt col</th><th>bal col</th><th>totals</th></tr>`;
    for (const r of rows) {
      const b = r.bank_statement;
      if (!b) continue;
      html += `<tr>
        <td>${esc(r.parser_id)}</td>
        <td>${b.n_transactions}</td>
        <td class="${cls(b.reconciliation_rate, .95, .7)}">${b.reconciliation_rate}</td>
        <td>${b.amount_column_consistency ?? "–"}${b.amount_columns_detected > 1 ? ` (${b.amount_columns_detected})` : ""}</td>
        <td>${b.balance_column_consistency ?? "–"}</td>
        <td class="${b.totals_match === true ? "good" : b.totals_match === false ? "bad" : ""}">${b.totals_match ?? "–"}</td>
      </tr>`;
    }
    html += `</table>`;

    if (rows.some((r) => r.ledger_score)) {
      html += `<h2>vs. ground truth</h2><table class="scores">
        <tr><th>parser</th><th>P</th><th>R</th><th>F1</th><th>desc</th><th>bal</th></tr>`;
      for (const r of rows) {
        const s = r.ledger_score;
        if (!s) continue;
        html += `<tr><td>${esc(r.parser_id)}</td>
          <td class="${cls(s.precision, .95, .8)}">${s.precision}</td>
          <td class="${cls(s.recall, .95, .8)}">${s.recall}</td>
          <td class="${cls(s.f1, .95, .8)}">${s.f1}</td>
          <td>${s.description_similarity ?? "–"}</td>
          <td>${s.balance_accuracy}</td></tr>`;
      }
      html += `</table>`;
    }

    const breaks = rows.flatMap((r) => (r.bank_statement?.breaks || []).slice(0, 3).map((b) => ({ ...b, parser: r.parser_id })));
    if (breaks.length) {
      html += `<h2>Reconciliation breaks</h2>`;
      for (const b of breaks) {
        html += `<div class="hunk"><div class="muted">${esc(b.parser)} · page ${b.page} · expected Δ ${b.expected_delta}, read ${b.amount}</div><div class="mono">${esc(b.raw)}</div></div>`;
      }
    }
  }

  const sim = state.scores.similarity;
  const ids = Object.keys(sim);
  if (ids.length > 1) {
    html += `<h2>Text agreement</h2><table class="scores"><tr><th></th>${ids.map((i) => `<th>${esc(i.slice(0, 6))}</th>`).join("")}</tr>`;
    for (const a of ids) {
      html += `<tr><td>${esc(a)}</td>${ids.map((b) => `<td class="${a === b ? "" : cls(sim[a][b], .95, .8)}">${sim[a][b].toFixed(2)}</td>`).join("")}</tr>`;
    }
    html += `</table>`;
  }
  host.innerHTML = html;
}

async function renderDiff() {
  const host = $("#tab-diff");
  const { a, b } = state.panes;
  if (!a.key || !b.key || a.key === b.key) {
    host.innerHTML = `<div class="muted">Pick two different parsers (turn on compare mode) to diff them.</div>`;
    return;
  }
  const out = await api(`/api/documents/${state.doc.doc_id}/diff`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ left: a.key, right: b.key }),
  });
  const left = state.results[a.key].parser_id;
  const right = state.results[b.key].parser_id;
  const note = out.same_content_different_order
    ? `<div class="muted" style="margin-top:10px">Same lines recovered — these parsers differ in serialization order, not content.</div>`
    : `<div class="muted" style="margin-top:10px">Identical line by line.</div>`;
  host.innerHTML =
    `<div class="muted">raw ${out.text_similarity.toFixed(3)} · lines ${out.line_similarity.toFixed(3)} · CER ${out.cer} · <span class="left">${esc(left)}</span> vs <span class="right">${esc(right)}</span></div>` +
    (out.hunks.length
      ? out.hunks.map((h) => `<div class="hunk"><div class="mono left">${h.left.map(esc).join("<br>") || "—"}</div><div class="mono right">${h.right.map(esc).join("<br>") || "—"}</div></div>`).join("")
      : note);
}

/* ---------------- wiring ---------------- */

function init() {
  $("#doc-select").addEventListener("change", (e) => e.target.value && selectDocument(e.target.value));
  $("#upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    status("uploading…");
    const meta = await api("/api/documents", { method: "POST", body: form });
    status("");
    await loadDocuments(meta.doc_id);
  });
  $("#prev-page").addEventListener("click", () => { if (state.page > 1) { state.page--; renderPage(); renderTab(); } });
  $("#next-page").addEventListener("click", () => { if (state.page < state.doc.pages) { state.page++; renderPage(); renderTab(); } });
  $("#zoom").addEventListener("input", (e) => { state.zoom = parseFloat(e.target.value); renderPage(); });
  $("#show-labels").addEventListener("change", renderPage);
  $("#compare-mode").addEventListener("change", (e) => {
    $("#pane-b").classList.toggle("hidden", !e.target.checked);
    renderPage();
  });
  $("#run-selected").addEventListener("click", runSelected);
  $$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $$(".result-select").forEach((sel) =>
    sel.addEventListener("change", () => {
      state.panes[sel.dataset.pane].key = sel.value;
      renderLayerChips();
      renderPage();
      renderTab();
    })
  );
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.key === "ArrowLeft") $("#prev-page").click();
    if (e.key === "ArrowRight") $("#next-page").click();
  });

  loadParsers().then(loadDocuments).catch((err) => status(err.message, true));
}

init();
