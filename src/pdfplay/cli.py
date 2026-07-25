"""Command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import registry
from .metrics import bank_statement, generic
from .models import ParseResult
from .runner import run_many
from .workspace import Workspace


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace(args.workspace)


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_options(pairs: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--opt expects key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        out[key] = _coerce(value)
    return out


def _parse_pages(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        elif part:
            pages.append(int(part))
    return pages or None


def _resolve_doc(ws: Workspace, doc_id: str) -> str:
    """Accept a full id, a unique prefix, or a filename substring."""
    docs = ws.list_documents()
    exact = [d for d in docs if d.doc_id == doc_id]
    if exact:
        return exact[0].doc_id
    matches = [d for d in docs if d.doc_id.startswith(doc_id) or doc_id.lower() in d.name.lower()]
    if len(matches) == 1:
        return matches[0].doc_id
    if not matches:
        raise SystemExit(f"no document matching {doc_id!r} (try `pdfplay docs`)")
    names = ", ".join(f"{d.doc_id[:8]}={d.name}" for d in matches)
    raise SystemExit(f"{doc_id!r} is ambiguous: {names}")


# -- commands ---------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    for spec in registry.describe_all():
        mark = "✓" if spec["available"] else "·"
        detail = spec["version"] if spec["available"] else spec["unavailable_reason"]
        print(f"{mark} {spec['id']:<13} {spec['kind']:<7} {detail}")
        if args.verbose:
            print(f"    {spec['description']}")
            if spec["options"]:
                opts = ", ".join(f"{o['name']}={o['default']}" for o in spec["options"])
                print(f"    options: {opts}")
    return 0


def cmd_samples(args: argparse.Namespace) -> int:
    from .samples import make_all

    for doc in make_all(args.out, seed=args.seed):
        ledger = doc.write_ledger()
        print(f"{doc.path}  ({len(doc.ledger['transactions'])} transactions, ledger: {ledger.name})")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    meta = ws.add_document(args.file, doc_class=args.doc_class)
    if args.ledger:
        ws.set_ground_truth(meta.doc_id, json.loads(Path(args.ledger).read_text()))
    print(f"{meta.doc_id}  {meta.name}  {meta.pages} page(s){'  [ledger]' if args.ledger else ''}")
    return 0


def cmd_docs(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    docs = ws.list_documents()
    if not docs:
        print("no documents yet — `pdfplay add FILE`")
        return 0
    for meta in docs:
        results = ws.list_results(meta.doc_id)
        gt = " [ledger]" if ws.get_ground_truth(meta.doc_id) else ""
        print(
            f"{meta.doc_id}  {meta.name:<40} {meta.pages:>3}p  "
            f"{meta.doc_class or '-':<16} {len(results)} result(s){gt}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    doc_id = _resolve_doc(ws, args.doc_id)
    if args.all:
        parser_ids = [p.id for p in registry.available_parsers()]
    else:
        parser_ids = args.parser
    if not parser_ids:
        raise SystemExit("pick parsers with -p/--parser, or use --all")

    options = {pid: _parse_options(args.opt) for pid in parser_ids}
    runs = run_many(
        ws, doc_id, parser_ids, _parse_pages(args.pages), options, force=args.force, max_workers=args.jobs
    )
    for result, key, cached in runs:
        tag = "cached" if cached else f"{result.duration_s:.2f}s"
        if result.status != "ok":
            print(f"✗ {result.parser_id:<13} {result.error}")
        else:
            blocks = len(result.all_blocks())
            print(f"✓ {result.parser_id:<13} {tag:<8} {len(result.pages)}p  {blocks} blocks  key={key}")
    return 0


def _rows(ws: Workspace, doc_id: str, doc_class: str) -> list[dict[str, Any]]:
    ledger = (ws.get_ground_truth(doc_id) or {}).get("transactions")
    rows = []
    for meta in ws.list_results(doc_id):
        result: ParseResult | None = ws.load_result(doc_id, meta["key"])
        if result is None:
            continue
        row = generic.analyze(result)
        row["key"] = meta["key"]
        if doc_class == "bank_statement" and result.status == "ok":
            row["bank_statement"] = bank_statement.analyze(result).as_dict()
            if ledger:
                row["ledger_score"] = bank_statement.score_against_ledger(result, ledger)
        rows.append(row)
    return rows


def cmd_score(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    doc_id = _resolve_doc(ws, args.doc_id)
    doc_class = args.doc_class or ws.get_document(doc_id).doc_class
    rows = _rows(ws, doc_id, doc_class)
    if not rows:
        print("no results yet — `pdfplay run DOC_ID --all`")
        return 1

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{'parser':<13} {'s/pg':>6} {'chars':>7} {'lines':>6} {'cover':>6} {'order':>6} {'cost':>8}")
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row['parser_id']:<13} ERROR  {row.get('error', '')[:60]}")
            continue
        cost = f"${row['cost_usd']:.4f}" if row.get("cost_usd") else "-"
        print(
            f"{row['parser_id']:<13} {row['seconds_per_page']:>6.2f} {row['n_chars']:>7} "
            f"{row['n_lines']:>6} {row['page_coverage']:>6} {row['reading_order_score']:>6} {cost:>8}"
        )

    if doc_class == "bank_statement":
        print(f"\n{'parser':<13} {'txns':>5} {'recon':>6} {'amt col':>8} {'bal col':>8} {'totals':>7}")
        for row in rows:
            b = row.get("bank_statement")
            if not b:
                continue
            print(
                f"{row['parser_id']:<13} {b['n_transactions']:>5} {b['reconciliation_rate']:>6.2f} "
                f"{str(b['amount_column_consistency']):>8} {str(b['balance_column_consistency']):>8} "
                f"{str(b['totals_match']):>7}"
            )
        if any(row.get("ledger_score") for row in rows):
            print(f"\n{'parser':<13} {'P':>6} {'R':>6} {'F1':>6} {'desc':>6} {'bal':>6}")
            for row in rows:
                s = row.get("ledger_score")
                if not s:
                    continue
                desc = s["description_similarity"]
                print(
                    f"{row['parser_id']:<13} {s['precision']:>6.2f} {s['recall']:>6.2f} {s['f1']:>6.2f} "
                    f"{(desc if desc is not None else 0):>6.2f} {s['balance_accuracy']:>6.2f}"
                )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    ws = _workspace(args)
    doc_id = _resolve_doc(ws, args.doc_id)
    results = {}
    for meta in ws.list_results(doc_id):
        result = ws.load_result(doc_id, meta["key"])
        if result and result.parser_id in args.parser:
            results[result.parser_id] = result
    missing = [p for p in args.parser if p not in results]
    if missing:
        raise SystemExit(f"no results for {', '.join(missing)} — run them first")

    left, right = (results[p] for p in args.parser[:2])
    report = generic.compare(left, right, limit=args.limit)
    print(
        f"raw text similarity {report['text_similarity']:.4f}   "
        f"line similarity {report['line_similarity']:.4f}   CER {report['cer']}"
    )
    if report["same_content_different_order"]:
        print("note: the same lines were recovered — the parsers differ in serialization order, not content")
    for hunk in report["hunks"]:
        print(f"\n[{hunk['tag']}]")
        for line in hunk["left"]:
            print(f"  - {line}")
        for line in hunk["right"]:
            print(f"  + {line}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    app = create_app(Workspace(args.workspace))
    print(f"pdfplay → http://{args.host}:{args.port}  (workspace: {Path(args.workspace).resolve()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


# -- entry point ------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pdfplay", description=__doc__)
    parser.add_argument("--workspace", default="workspace", help="workspace directory (default: ./workspace)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list parsers and availability")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("samples", help="generate synthetic bank statements with ledgers")
    p.add_argument("--out", default="samples")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("add", help="add a PDF to the workspace")
    p.add_argument("file")
    p.add_argument("--class", dest="doc_class", default="", help="e.g. bank_statement")
    p.add_argument("--ledger", help="ground-truth JSON to attach")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("docs", help="list documents in the workspace")
    p.set_defaults(func=cmd_docs)

    p = sub.add_parser("run", help="run parsers over a document")
    p.add_argument("doc_id")
    p.add_argument("-p", "--parser", action="append", default=[])
    p.add_argument("--all", action="store_true", help="every available parser")
    p.add_argument("--pages", help="e.g. 1,3 or 1-4")
    p.add_argument("--opt", action="append", help="parser option, key=value (repeatable)")
    p.add_argument("--force", action="store_true", help="ignore cached results")
    p.add_argument("--jobs", type=int, default=4)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score every stored result for a document")
    p.add_argument("doc_id")
    p.add_argument("--class", dest="doc_class", default="", help="override the document class")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("compare", help="line-level diff between two parsers")
    p.add_argument("doc_id")
    p.add_argument("-p", "--parser", action="append", default=[], required=True)
    p.add_argument("--limit", type=int, default=40)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("serve", help="run the web viewer")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
