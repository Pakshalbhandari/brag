"""CLI for the DOCX/PPTX/XLSX RAG pipeline.

    brag parse samples/report.docx      # preview chunks, no API calls
    brag ingest samples/                # parse + index files/folders
    brag query "What drove Q3 growth?"  # one question
    brag chat                           # interactive Q&A
"""

import argparse
import asyncio
import json
import logging

from dotenv import find_dotenv, load_dotenv

from . import __version__

MODES = ["mix", "hybrid", "local", "global", "naive"]


def cmd_parse(args: argparse.Namespace) -> None:
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


async def cmd_ingest(args: argparse.Namespace) -> None:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        results = await pipeline.ingest(args.paths)
        for r in results:
            print(f"Indexed {r.source.name}: {r.stats}")
    finally:
        await pipeline.close()


async def cmd_query(args: argparse.Namespace) -> None:
    from .pipeline import RAGPipeline

    pipeline = RAGPipeline()
    try:
        print(await pipeline.query(args.question, mode=args.mode))
    finally:
        await pipeline.close()


async def cmd_chat(args: argparse.Namespace) -> None:
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


def main() -> None:
    ap = argparse.ArgumentParser(prog="brag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("parse", help="Parse and chunk documents without indexing")
    p.add_argument("paths", nargs="+")
    p.add_argument("--json", action="store_true", help="Print the full RAG-Anything content list")

    p = sub.add_parser("ingest", help="Parse and index documents")
    p.add_argument("paths", nargs="+")

    p = sub.add_parser("query", help="Ask one question")
    p.add_argument("question")
    p.add_argument("--mode", choices=MODES, default="mix")

    p = sub.add_parser("chat", help="Interactive Q&A")
    p.add_argument("--mode", choices=MODES, default="mix")

    args = ap.parse_args()
    # Look for .env from the current directory upwards, not from the installed package's location.
    load_dotenv(find_dotenv(usecwd=True))
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.command == "parse":
        cmd_parse(args)
    else:
        asyncio.run({"ingest": cmd_ingest, "query": cmd_query, "chat": cmd_chat}[args.command](args))


if __name__ == "__main__":
    main()
