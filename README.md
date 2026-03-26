# Local PDF RAG

Local Retrieval-Augmented Generation (RAG) pipeline for ingesting and querying PDF documents using Ollama.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

---
### Streamlit run

```bash
streamlit run streamlit_app.py
```

The app opens in your browser at `http://localhost:8501`.

**Workflow:**

1. Upload PDFs - add files in the sidebar 
2. Ingest uploaded files - update the vector store.
3. Chat - enter questions about pdf content.
4. Sessions - load or start new sessions from the sidebar.

---
### Command line run

#### 1. Ingest PDFs

Place PDF documents inside `data/raw/` or supply a custom path and run:

```bash
python -m src.main ingest
# or provide a path
python -m src.main ingest /path/to/docs
```

This builds/updates a FAISS vector store persisted under `data/vectorstore/`.

#### 2. Ask questions

After ingestion, query the knowledge base:

```bash
python -m src.main chat "What is the main topic?"
```

The command prints the session identifier along with the generated answer. Use
`--session <id>` to continue the same conversation so that context is preserved.

#### 3. List sessions

```bash
python -m src.main sessions
```

## Project layout

```
streamlit_app.py  # Streamlit entry point
src/
  main.py         # CLI entry point
  ingest.py       # PDF ingestion pipeline
  chain.py        # Conversational retrieval chain
  utils.py        # Shared utilities
data/
  raw/            # Source PDFs
  vectorstore/    # Persisted FAISS index
  sessions/       # Stored chat histories
```

## Models setup

- Embeddings model: `sentence-transformers/all-MiniLM-L6-v2`

- LLM: Ollama llama3.2:1b model

  1. Install Ollama: https://ollama.com/download
  2. Pull model: `ollama pull llama3.2:1b`
