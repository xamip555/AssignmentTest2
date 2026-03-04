from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from database import (
    document_exists,
    get_max_vector_id,
    get_vector_ids_for_document,
    init_db,
    list_documents,
    delete_document,
)
from ingestion import FaissStore, extract_text_from_bytes, ingest_document
from retrieval import answer_with_citations

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "rag.db"
FAISS_PATH = BASE_DIR / "faiss.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

app = FastAPI(title="RAG App with Citations")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class AskRequest(BaseModel):
    question: str


@app.on_event("startup")
def startup() -> None:
    init_db(DB_PATH)

    max_vector_id = get_max_vector_id(DB_PATH)
    app.state.faiss_store = FaissStore.load_or_create(
        index_path=FAISS_PATH,
        dimension=EMBEDDING_DIM,
        start_id=max_vector_id + 1,
    )
    app.state.embedder = SentenceTransformer(EMBEDDING_MODEL)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    allowed = (".txt", ".md", ".pdf")
    lower_name = file.filename.lower()
    if not lower_name.endswith(allowed):
        raise HTTPException(status_code=400, detail="Only TXT, MD, and PDF files are supported.")

    raw_bytes = await file.read()
    text = extract_text_from_bytes(file.filename, raw_bytes)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file did not contain readable text.")

    try:
        result = ingest_document(
            file_name=file.filename,
            text=text,
            db_path=DB_PATH,
            faiss_store=app.state.faiss_store,
            embedder=app.state.embedder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "id": result["doc_id"],
        "name": result["name"],
        "chunks_indexed": result["chunks_indexed"],
    }


@app.get("/documents")
async def documents() -> list[dict]:
    return list_documents(DB_PATH)


@app.delete("/documents/{doc_id}")
async def remove_document(doc_id: int) -> dict:
    if not document_exists(doc_id, DB_PATH):
        raise HTTPException(status_code=404, detail="Document not found.")

    vector_ids = get_vector_ids_for_document(doc_id, DB_PATH)
    app.state.faiss_store.remove_ids(vector_ids)
    app.state.faiss_store.save()

    delete_document(doc_id, DB_PATH)
    return {"deleted": doc_id}


@app.post("/ask")
async def ask(payload: AskRequest) -> dict:
    try:
        return answer_with_citations(
            question=payload.question,
            db_path=DB_PATH,
            faiss_store=app.state.faiss_store,
            embedder=app.state.embedder,
            top_k=5,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
