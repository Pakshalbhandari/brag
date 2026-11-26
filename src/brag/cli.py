"""CLI for the DOCX/PPTX/XLSX RAG pipeline.

    brag parse samples/report.docx      # preview chunks, no API calls
    brag ontology propose samples/      # suggest entity types -> ontology.yaml
    brag ingest samples/                # index files/folders; re-run to pick up edits
    brag query "What drove Q3 growth?"  # one question
    brag chat                           # interactive Q&A
    brag list                           # indexed files
    brag remove samples/old.pptx        # drop files (or folders) from the index
    brag eval eval/samples.yaml         # retrieval metrics against a golden set
"""

import argparse
import asyncio
import json
import logging
import sys

from dotenv import find_dotenv, load_dotenv

from . import __version__

MODES = ["mix", "hybrid", "local", "global", "naive"]


def cmd_parse(args: argparse.Namespace) -> int:
    from .config import Settings
    from .parsing import CHUNK_SEPARATOR, DocumentParser
    from .pipeline import collect_files

    parser = DocumentParser(Settings())
    for f in collect_files(args.paths):
        parsed = parser.parse(f)
        print(f"\n== {f}  ({parsed.doc_id})  {parsed.stats}")
        if args.json:
            print(json.dumps(parsed.content_list, indent=2, ensure_ascii=False))
            continue
        for item in parsed.content_list:
            body = item.get("text") or item.get("table_body") or item.get("img_path") or item.get("latex", "")
            body = body.replace(CHUNK_SEPARATOR, "").rstrip()
            print(f"--- {item['type']} (page_idx={item.get('page_idx')})")
            print(body[:400] + (" …" if len(body) > 400 else ""))
    return 0


async def cmd_ontology_propose(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .config import Settings
    from .models import ClaudeModels
    from .ontology import Ontology, collect_samples, propose
    from .parsing import DocumentParser
    from .pipeline import collect_files

    settings = Settings()
    files = collect_files(args.paths)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1
    path = Path(settings.ontology_file)
    ontology = Ontology.load(path) if path.exists() else Ontology()

    print(f"Sampling {len(files)} file(s)...")
    samples = collect_samples(DocumentParser(settings), files)
    models = ClaudeModels(settings)
    proposed = await propose(models, samples)
    added = ontology.merge(proposed)
    ontology.save(path)

    if added:
        print(f"Added {len(added)} entity type(s) to {path}:")
        for t in added:
            print(f"  {t.name:<22} {t.description}")
    else:
        print(f"No new entity types; {path} already covers the proposal.")
    print(f"\n{len(ontology.entity_types)} types in total. Review and edit {path}, then run `brag ingest`.")
    print(models.usage.summary())
    return 0


async def cmd_ingest(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        results = await pipeline.ingest(args.paths)
    finally:
        await pipeline.close()

    if not results:
        print("No supported files found.", file=sys.stderr)
        return 1
    for r in results:
        detail = r.error if r.status == "failed" else (r.stats or "")
        print(f"{r.status:<10} {r.source.name}  {detail}".rstrip())
    failed = sum(r.status == "failed" for r in results)
    counts = {s: sum(r.status == s for r in results) for s in ("indexed", "updated", "unchanged", "empty", "failed")}
    print("\n" + ", ".join(f"{n} {s}" for s, n in counts.items() if n))
    if pipeline.models.usage.requests:
        print(pipeline.models.usage.summary())
    if failed:
        print(f"{failed} file(s) failed; run with -v for details.", file=sys.stderr)
    return 1 if failed else 0


async def cmd_remove(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        results = await pipeline.remove(args.paths)
    finally:
        await pipeline.close()
    if not results:
        print("None of these files are in the index.", file=sys.stderr)
        return 1
    for r in results:
        print(f"{'removed' if r.ok else 'failed':<8} {r.label}  {r.error or ''}".rstrip())
    return 0 if all(r.ok for r in results) else 1


def cmd_list(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .config import Settings
    from .manifest import Manifest

    manifest = Manifest.load(Settings().manifest_file)
    if not manifest.files:
        print("The index is empty.")
        return 0
    for key, entry in sorted(manifest.files.items(), key=lambda kv: kv[1].label):
        missing = "" if Path(key).exists() else "  (file no longer exists; `brag remove` it)"
        print(f"{entry.indexed_at}  {entry.label}{missing}")
    return 0


async def cmd_query(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        print(await pipeline.query(args.question, mode=args.mode))
    finally:
        await pipeline.close()
    return 0


async def cmd_chat(args: argparse.Namespace) -> int:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        while True:
            try:
                question = (await asyncio.to_thread(input, "\n? ")).strip()
            except EOFError:
                break
            if question.lower() in {"", "exit", "quit"}:
                break
            print(await pipeline.query(question, mode=args.mode))
    finally:
        await pipeline.close()
    return 0


def _fmt(value: float | None) -> str:
    return "  -  " if value is None else f"{value:.3f}"


async def cmd_eval(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .evaluation import load_golden, run_eval
    from .pipeline import RAGPipeline

    cases = load_golden(args.golden)
    pipeline = RAGPipeline()
    try:
        report = await run_eval(pipeline, cases, mode=args.mode, k=args.k, answers=args.answers)
    finally:
        await pipeline.close()

    print(f"{'case':<24} {'recall':>7} {'mrr':>7} {'facts':>7} {'ground':>7}")
    for r in report.results:
        if r.error:
            print(f"{r.id:<24} error: {r.error}")
            continue
        print(f"{r.id:<24} {_fmt(r.recall):>7} {_fmt(r.reciprocal_rank):>7} {_fmt(r.fact_recall):>7} {_fmt(r.groundedness):>7}")
        if r.missing_facts:
            print(f"{'':<24} missing facts: {', '.join(r.missing_facts)}")
        for claim in r.unsupported_claims:
            print(f"{'':<24} unsupported: {claim}")
    summary = report.summary
    print(
        f"\nmode={args.mode} k={args.k}  recall@k={_fmt(summary['recall@k'])}  hit@k={_fmt(summary['hit@k'])}"
        f"  mrr={_fmt(summary['mrr'])}  facts={_fmt(summary['facts'])}  groundedness={_fmt(summary['groundedness'])}"
    )
    if pipeline.models.usage.requests:
        print(pipeline.models.usage.summary())
    if args.output:
        Path(args.output).write_text(report.to_json(), encoding="utf-8")
        print(f"Report written to {args.output}")

    failures = report.failures({
        "recall@k": args.min_recall, "mrr": args.min_mrr,
        "facts": args.min_facts, "groundedness": args.min_groundedness,
    })
    for f in failures:
        print(f"FAIL {f}", file=sys.stderr)
    return 1 if failures else 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="brag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="Parse and chunk documents without indexing")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="Print the full RAG-Anything content list")

    p = sub.add_parser("ontology", help="Manage the knowledge graph's entity types")
    osub = p.add_subparsers(dest="ontology_command", required=True)
    op = osub.add_parser("propose", help="Suggest entity types from sample documents (adds to ontology.yaml)")
    op.add_argument("paths", nargs="+")

    p = sub.add_parser("ingest", help="Index documents; edited files replace their old version")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("remove", help="Remove files, or every file under folders, from the index")
    p.add_argument("paths", nargs="+")

    sub.add_parser("list", help="List indexed files")

    p = sub.add_parser("query", help="Ask one question")
    p.add_argument("question")
    p.add_argument("--mode", choices=MODES, default="mix")

    p = sub.add_parser("chat", help="Interactive Q&A")
    p.add_argument("--mode", choices=MODES, default="mix")

    p = sub.add_parser("eval", help="Score retrieval (and optionally answers) against a golden question set")
    p.add_argument("golden", help="YAML file with cases (see brag.evaluation)")
    p.add_argument("--mode", choices=MODES, default="mix")
    p.add_argument("-k", "--k", type=int, default=5, help="Chunks considered for recall@k and MRR (default 5)")
    p.add_argument("--answers", action="store_true", help="Also generate answers and score facts and groundedness")
    p.add_argument("--output", help="Write the full report as JSON")
    for metric in ("recall", "mrr", "facts", "groundedness"):
        p.add_argument(f"--min-{metric}", type=float, help=f"Exit 1 if mean {metric} is below this")

    args = ap.parse_args()
    # Look for .env from the current directory upwards, not from the installed package's location.
    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "parse":
        code = cmd_parse(args)
    elif args.command == "list":
        code = cmd_list(args)
    elif args.command == "ontology":
        code = asyncio.run(cmd_ontology_propose(args))
    else:
        handlers = {"ingest": cmd_ingest, "remove": cmd_remove, "query": cmd_query, "chat": cmd_chat, "eval": cmd_eval}
        code = asyncio.run(handlers[args.command](args))
    sys.exit(code)


if __name__ == "__main__":
    main()
