"""Retrieval + generation chain for answering questions over ingested data."""

from __future__ import annotations

from typing import Optional

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama

from .utils import (
    append_to_session,
    generate_session_id,
    get_embeddings,
    load_session_history,
    load_vectorstore,
)


def build_chain(model: str = "llama3.2:1b") -> ConversationalRetrievalChain:
    embeddings = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    if not vectorstore:
        raise RuntimeError("Vector store not found. Please run ingest first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=model)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )


def chat(
    question: str,
    chain: Optional[ConversationalRetrievalChain] = None,
    session_id: Optional[str] = None,
) -> tuple[str, str]:
    """Ask a question using the conversational retrieval chain.

    Returns a tuple of (session_id, answer).
    """

    if chain is None:
        chain = build_chain()

    if session_id is None:
        session_id = generate_session_id()

    history = load_session_history(session_id)
    # Preload memory with existing history
    for user, assistant in history:
        chain.memory.chat_memory.add_user_message(user)
        chain.memory.chat_memory.add_ai_message(assistant)

    result = chain({"question": question})
    answer = result["answer"]

    append_to_session(session_id, question, answer)
    return session_id, answer

