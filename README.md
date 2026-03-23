# myRAG

Local RAG pipeline using FAISS and Ollama.

## Quick start

Create a virtual environment (Python 3.10+) and install dependencies:

```bash
pip install -r requirements.txt
```

### 1. Ingest PDFs

Place PDF documents inside `data/raw/` or supply a custom path and run:

```bash
python -m src.main ingest
# or provide a path
python -m src.main ingest /path/to/docs
```

This builds/updates a FAISS vector store persisted under `data/vectorstore/`.

### 2. Ask questions

After ingestion, query the knowledge base:

```bash
python -m src.main chat "What is the main topic?"
```

The command prints the session identifier along with the generated answer. Use
`--session <id>` to continue the same conversation so that context is preserved.

### 3. List sessions

```bash
python -m src.main sessions
```

## Project layout

```
src/
  main.py      # CLI entry point
  ingest.py    # Offline ingestion pipeline
  chain.py     # Conversational retrieval chain
  utils.py     # Shared utilities (embeddings, storage, sessions)
data/
  raw/         # Source PDFs
  vectorstore/ # Persisted FAISS index
  sessions/    # Stored chat histories
```

## Environment variables

- `EMBEDDINGS_MODEL_NAME` - override default `sentence-transformers/all-MiniLM-L6-v2`
  embeddings model.
- `OLLAMA_BASE_URL` - point LangChain to a remote/local Ollama server if needed.

## Notes

- Ensure Ollama is running with a compatible chat model (default `mistral`).
- Add `data/vectorstore` and `data/sessions` to `.gitignore` for cleanliness in VCS.
