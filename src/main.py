from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Sequence

from .chain import chat
from .ingest import ingest
from .utils import ensure_directories, list_sessions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest PDF documents into the vector store"
    )
    ingest_parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional path to a PDF file or directory with PDFs",
    )

    chat_parser = subparsers.add_parser(
        "chat", help="Ask a question using the RAG pipeline"
    )
    chat_parser.add_argument("question", help="Question to ask the assistant")
    chat_parser.add_argument(
        "--session",
        dest="session_id",
        default=None,
        help="Existing session identifier to continue the conversation",
    )

    subparsers.add_parser("sessions", help="List available session identifiers")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    ensure_directories()

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        path = Path(args.path).resolve() if args.path else None
        added = ingest(path)
        print(f"Ingestion complete. Added {added} chunks to the vector store.")
        return 0

    if args.command == "chat":
        session_id, answer = chat(args.question, session_id=args.session_id)
        print(f"Session: {session_id}\nAnswer: {answer}")
        return 0

    if args.command == "sessions":
        sessions = list(list_sessions())
        if not sessions:
            print("No sessions found.")
        else:
            print("Available sessions:")
            for item in sessions:
                print(f" - {item}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())

