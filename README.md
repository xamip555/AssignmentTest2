# RAG App with Citations

A small Retrieval-Augmented Generation (RAG) web app built with FastAPI.

You can:
- Upload `.txt`, `.md`, or `.pdf` files
- Ask questions against uploaded content
- Get an answer with citations and inspect the retrieved source chunks
- Delete documents from the knowledge base

The app stores metadata/chunks in SQLite and vector embeddings in a FAISS index.

## Tech Stack

- Backend: FastAPI
- Frontend: Jinja2 template + vanilla JavaScript
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- Vector search: `faiss-cpu`
- Metadata/chunk store: SQLite
- Optional answer generation: OpenAI Responses API

## Project Structure

```text
rag_app/
  main.py            # FastAPI app + routes
  ingestion.py       # file parsing, chunking, embedding, FAISS writes
  retrieval.py       # retrieval + answer generation + citations
  database.py        # SQLite schema and queries
  templates/index.html
  static/script.js
  requirements.txt
  rag.db             # SQLite database (generated/updated at runtime)
  faiss.index        # FAISS index (generated/updated at runtime)
```

## How It Works

1. Document is uploaded via `/upload`.
2. Text is extracted (`pypdf` for PDFs, decoding for text files).
3. Text is chunked (default: 500 chars, 100 overlap).
4. Chunks are embedded with `all-MiniLM-L6-v2` and stored in FAISS with vector IDs.
5. Chunk metadata + text are stored in SQLite.
6. On `/ask`, query is embedded and top-k chunks are retrieved.
7. Answer is generated:
   - with OpenAI (if `OPENAI_API_KEY` exists), or
   - with local extractive fallback logic.
8. Response includes `answer`, `citations`, and full `sources`.

## Setup

### 1) Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r rag_app/requirements.txt
```

### 3) (Optional) Configure OpenAI

If you want LLM-based answers instead of local fallback:

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
$env:OPENAI_MODEL="gpt-4.1-mini"
```

macOS/Linux:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4.1-mini"
```

If `OPENAI_API_KEY` is not set, the app still works with local answer generation.

### 4) Run the server

From project root:

```bash
uvicorn rag_app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## API Endpoints

- `GET /` : Web UI
- `POST /upload` : Upload one document (`.txt`, `.md`, `.pdf`)
- `GET /documents` : List uploaded documents
- `DELETE /documents/{doc_id}` : Delete a document and its vectors
- `POST /ask` : Ask a question

Example `/ask` request body:

```json
{
  "question": "What is this document about?"
}
```

Example response shape:

```json
{
  "answer": "...",
  "citations": [
    {
      "doc_id": 1,
      "doc_name": "sample.pdf",
      "chunk_id": 10,
      "snippet": "..."
    }
  ],
  "sources": [
    {
      "doc_id": 1,
      "doc_name": "sample.pdf",
      "chunk_id": 10,
      "chunk_index": 2,
      "score": 0.83,
      "text": "..."
    }
  ]
}
```

## Notes and Limitations

- Supported upload types: `.txt`, `.md`, `.pdf`
- Chunking is character-based (not token-based)
- Current FAISS index type is `IndexFlatIP` with normalized embeddings (cosine similarity behavior)
- `rag.db` and `faiss.index` are local files; back them up if needed
- Deleting a document removes both DB rows and related vector IDs from FAISS
- No authentication is implemented (intended for local/dev usage)

## Quick Dev Tips

- To reset data, stop server and remove `rag_app/rag.db` and `rag_app/faiss.index`
- Keep `sentence-transformers` model download in mind on first run (can take time)

## Future Improvements

- Token-aware chunking and smarter retriever settings
- Better citation ranking/filtering
- Streaming responses in UI
- Auth + multi-user support
- Docker setup


## Demo video:

https://github.com/user-attachments/assets/0a426e8a-a1bb-477e-a209-a42cfdd76320

