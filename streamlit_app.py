"""Streamlit web UI for the Local PDF RAG application.

Run from the project root:
    streamlit run streamlit_app.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from src.chain import build_chain, chat
from src.ingest import ingest
from src.utils import (
    ensure_directories,
    list_sessions,
    load_session_history,
    vectorstore_exists,
)

st.set_page_config(
    page_title="Local PDF RAG",
    page_icon="📄",
    layout="wide",
)

ensure_directories()


@st.cache_resource(show_spinner="Loading language model…")
def get_chain():
    """Build and cache the conversational retrieval chain."""
    return build_chain()


if "session_id" not in st.session_state:
    st.session_state["session_id"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []


with st.sidebar:
    st.title("📄 Local PDF RAG")
    st.divider()

    st.subheader("1 · Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.write("Files ready for ingestion:")
        for f in uploaded_files:
            st.text(f"- {f.name}")

    if st.button("Ingest uploaded files", disabled=not uploaded_files):
        with st.spinner("Ingesting…"):
            try:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    for uploaded in uploaded_files:
                        dest = tmp_path / uploaded.name
                        dest.write_bytes(uploaded.getvalue())
                    added = ingest(tmp_path)
                get_chain.clear()
                st.success(f"✅ Added {added} chunks from {len(uploaded_files)} file(s).")
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()

    st.subheader("2 · Sessions")
    existing_sessions = sorted(list(list_sessions()))
    session_options = ["— new session —"] + existing_sessions
    selected_option = st.selectbox(
        "Load a session",
        options=session_options,
        index=0,
        label_visibility="collapsed",
    )

    if st.button("Load session"):
        if selected_option == "— new session —":
            st.session_state["session_id"] = None
            st.session_state["messages"] = []
        else:
            sid = selected_option
            history = load_session_history(sid)
            st.session_state["session_id"] = sid
            st.session_state["messages"] = [
                msg
                for u, a in history
                for msg in (
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                )
            ]
        st.rerun()

    if st.session_state["session_id"]:
        st.caption(f"Active session: `{st.session_state['session_id']}`")
    else:
        st.caption("Active session: *new*")


st.header("Chat")

if not vectorstore_exists():
    st.warning(
        "⚠️ No vector store found. Upload and ingest at least one PDF in the sidebar first."
    )

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask a question about your documents…",
    disabled=not vectorstore_exists(),
)

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                chain = get_chain()
                session_id, answer = chat(
                    user_input,
                    chain=chain,
                    session_id=st.session_state["session_id"],
                )
                st.session_state["session_id"] = session_id
            except Exception as exc:
                answer = f"Error: {exc}"

        st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
