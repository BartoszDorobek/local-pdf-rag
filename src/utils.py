from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SESSIONS_DIR = DATA_DIR / "sessions"

DEFAULT_EMBEDDINGS_MODEL = os.getenv(
    "EMBEDDINGS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


def ensure_directories() -> None:
    """Create required directory structure if it does not already exist."""

    for directory in (DATA_DIR, RAW_DATA_DIR, VECTORSTORE_DIR, SESSIONS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Return a shared embeddings instance for the application."""

    embeddings_model = model_name or DEFAULT_EMBEDDINGS_MODEL
    return HuggingFaceEmbeddings(model_name=embeddings_model)


def vectorstore_exists(persist_dir: Path = VECTORSTORE_DIR) -> bool:
    """Check whether a FAISS vector store has been persisted in the directory."""

    index_file = persist_dir / "index.faiss"
    metadata_file = persist_dir / "index.pkl"
    return index_file.exists() and metadata_file.exists()


def load_vectorstore(
    embeddings: Optional[HuggingFaceEmbeddings] = None,
    persist_dir: Path = VECTORSTORE_DIR,
) -> Optional[FAISS]:
    """Load a persisted FAISS vector store if available."""

    if not vectorstore_exists(persist_dir):
        return None

    embeddings = embeddings or get_embeddings()
    return FAISS.load_local(
        str(persist_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def save_vectorstore(store: FAISS, persist_dir: Path = VECTORSTORE_DIR) -> None:
    """Persist a FAISS vector store to disk."""

    persist_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(persist_dir))


def generate_session_id() -> str:
    """Generate a unique session identifier."""

    return uuid.uuid4().hex


def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def load_session_history(session_id: str) -> List[Tuple[str, str]]:
    """Load chat history as a list of (user, assistant) turns."""

    ensure_directories()
    path = _session_path(session_id)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        payload: Sequence[Sequence[str]] = json.load(file)

    history: List[Tuple[str, str]] = []
    for turn in payload:
        if (
            isinstance(turn, Sequence)
            and len(turn) == 2
            and all(isinstance(item, str) for item in turn)
        ):
            history.append((turn[0], turn[1]))
    return history


def append_to_session(session_id: str, question: str, answer: str) -> None:
    """Persist a single conversational turn for the session."""

    ensure_directories()
    history = load_session_history(session_id)
    history.append((question, answer))
    path = _session_path(session_id)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)


def list_sessions() -> Iterable[str]:
    """Return existing session identifiers available on disk."""

    ensure_directories()
    for file in SESSIONS_DIR.glob("*.json"):
        yield file.stem

