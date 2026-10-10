"""Command line entry point: `intake <command>`.

    intake process <folder>             run one dossier and print the decision and findings
    intake index                        build the policy search index under var/index
    intake ask "<question>"             answer a question from the policy documents
    intake eval idp [--model M]         score the pipeline on fixtures/dossiers
    intake eval rag [--model M]         score retrieval and answers on eval/rag_questions.jsonl
    intake eval retrieval               compare embedding-only and hybrid retrieval (no model calls)
    intake costs                        model spend recorded in the journal
    intake serve                        start the HTTP API
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from loan_intake.pipeline import DEFAULT_CLASSIFY_MODEL, DEFAULT_EXTRACT_MODEL

ROOT = Path(__file__).resolve().parents[2]
VAR = ROOT / "var"
INDEX_DIR = VAR / "index"
DB_PATH = VAR / "intake.db"
POLICY_DOCS = ROOT / "policy"


def _index(rebuild: bool = False):
    from loan_intake.rag.chunking import load_corpus
    from loan_intake.rag.index import Embedder, MatrixStore

    embedder = Embedder()
    if rebuild or not (INDEX_DIR / "chunks.json").exists():
        store = MatrixStore.build(load_corpus(POLICY_DOCS), embedder)
        store.save(INDEX_DIR)
    else:
        store = MatrixStore.load(INDEX_DIR)
    return store, embedder


def main(argv: list[str] | None = None) -> int:
    load_dotenv(find_dotenv(usecwd=True))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="intake")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("process")
    p.add_argument("folder", type=Path)
    p.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    sub.add_parser("index")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    e = sub.add_parser("eval")
    e.add_argument("suite", choices=["idp", "rag", "retrieval"])
    e.add_argument("--model", default=DEFAULT_EXTRACT_MODEL)
    e.add_argument("--classify-model", default=DEFAULT_CLASSIFY_MODEL)
    e.add_argument("--limit", type=int)
    e.add_argument("--subset", choices=["standard", "stress"], help="IDP only: run one fixture set")
    e.add_argument("--no-cache", action="store_true", help="always call the API, ignoring var/cache")
    sub.add_parser("costs")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "process":
        from loan_intake.documents import load_dir
        from loan_intake.llm import AnthropicClient
        from loan_intake.pipeline import Settings, process
        from loan_intake.store import Store

        result = process(args.folder.name, load_dir(args.folder), Settings(AnthropicClient(), args.model))
        Store(DB_PATH).save(result)
        print(f"{result.dossier_id}: {result.decision.value}")
        for f in result.findings:
            print(f"  [{f.severity.value}] {f.code} {f.document} {f.field}: {f.message}")
        print(json.dumps(result.metrics, ensure_ascii=False))
    elif args.command == "index":
        store, _ = _index(rebuild=True)
        print(f"indexed {len(store.chunks)} sections into {INDEX_DIR}")
    elif args.command == "ask":
        from loan_intake.llm import AnthropicClient
        from loan_intake.rag.answer import ask

        store, embedder = _index()
        answer = ask(args.question, store, embedder, AnthropicClient(), args.model)
        print(answer.text)
        print("sources:", ", ".join(answer.citations) or "-")
    elif args.command == "eval":
        from loan_intake.evaluation import report
        from loan_intake.llm import AnthropicClient

        if args.suite == "retrieval":
            from loan_intake.evaluation import rag

            store, embedder = _index()
            result = rag.retrieval_sweep(store, embedder)
            for row in result["rows"]:
                print(row)
            print(f"report: {report.write('retrieval', result).relative_to(ROOT)}")
            return 0
        from loan_intake.llm import CachingClient

        client = AnthropicClient() if args.no_cache else CachingClient(AnthropicClient(), VAR / "cache")
        if args.suite == "idp":
            from loan_intake.evaluation import idp
            from loan_intake.pipeline import Settings

            result = idp.run(Settings(client, args.model, args.classify_model), limit=args.limit, subset=args.subset)
        else:
            from loan_intake.evaluation import rag

            store, embedder = _index()
            result = rag.run(store, embedder, client, args.model)
        path = report.write(args.suite, result)
        print(json.dumps(result["summary"], indent=1))
        print(f"report: {path.relative_to(ROOT)}")
    elif args.command == "costs":
        from loan_intake.store import Store

        for row in Store(DB_PATH).costs():
            print(row)
    elif args.command == "serve":
        import uvicorn

        uvicorn.run("loan_intake.api:app", host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
