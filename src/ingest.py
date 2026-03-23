from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .utils import (
    RAW_DATA_DIR,
    ensure_directories,
    get_embeddings,
    load_vectorstore,
    save_vectorstore,
)


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _load_documents_from_path(path: Path) -> Sequence[Document]:
    if path.is_dir():
        documents: list[Document] = []
        for pdf_path in sorted(path.glob("*.pdf")):
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
        return documents

    if path.is_file() and path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    raise ValueError(f"Unsupported path supplied for ingestion: {path}")


def ingest(source: Path | None = None) -> int:
    """Ingest PDF documents into the FAISS vector store.

    Returns the number of chunks added to the store.
    """

    ensure_directories()
    source_path = source or RAW_DATA_DIR

    documents = _load_documents_from_path(source_path)
    if not documents:
        raise ValueError(f"No PDF documents found in {source_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = get_embeddings()
    store = load_vectorstore(embeddings)
    if store:
        store.add_documents(chunks)
    else:
        store = FAISS.from_documents(chunks, embeddings)

    save_vectorstore(store)
    return len(chunks)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(argv or sys.argv[1:])
    source = Path(argv[0]) if argv else None
    added = ingest(source)
    print(f"Ingestion complete. Added {added} chunks to the vector store.")


if __name__ == "__main__":
    main()

