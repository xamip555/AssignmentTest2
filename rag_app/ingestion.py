from __future__ import annotations

import io
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from database import add_chunks, create_document


class FaissStore:
    def __init__(self, index_path: Path, index: faiss.IndexIDMap2, start_id: int) -> None:
        self.index_path = index_path
        self.index = index
        self.next_vector_id = start_id
        self._lock = threading.Lock()

    @classmethod
    def load_or_create(cls, index_path: Path, dimension: int, start_id: int = 1) -> "FaissStore":
        if index_path.exists():
            loaded = faiss.read_index(str(index_path))
            if not isinstance(loaded, faiss.IndexIDMap2):
                wrapped = faiss.IndexIDMap2(loaded)
                return cls(index_path=index_path, index=wrapped, start_id=start_id)
            return cls(index_path=index_path, index=loaded, start_id=start_id)

        base_index = faiss.IndexFlatIP(dimension)
        id_map = faiss.IndexIDMap2(base_index)
        return cls(index_path=index_path, index=id_map, start_id=start_id)

    def add_embeddings(self, embeddings: np.ndarray) -> list[int]:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        with self._lock:
            vector_ids = np.arange(
                self.next_vector_id,
                self.next_vector_id + len(embeddings),
                dtype=np.int64,
            )
            self.index.add_with_ids(embeddings, vector_ids)
            self.next_vector_id += len(embeddings)
            return vector_ids.tolist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")
        return self.index.search(query_embedding, top_k)

    def remove_ids(self, vector_ids: list[int]) -> None:
        if not vector_ids:
            return
        remove_array = np.array(vector_ids, dtype=np.int64)
        self.index.remove_ids(remove_array)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))


def split_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - overlap)

    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


def extract_text_from_bytes(file_name: str, data: bytes) -> str:
    lower_name = file_name.lower()
    if lower_name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    for encoding in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="ignore")


def ingest_document(
    file_name: str,
    text: str,
    db_path: Path,
    faiss_store: FaissStore,
    embedder,
) -> dict:
    chunks = split_text(text)
    if not chunks:
        raise ValueError("The file has no usable text after processing.")

    embeddings = embedder.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    vector_ids = faiss_store.add_embeddings(embeddings)
    doc_id = create_document(file_name, db_path=db_path)

    chunk_records = [
        (chunk_index, chunk_text, vector_id)
        for chunk_index, (chunk_text, vector_id) in enumerate(zip(chunks, vector_ids))
    ]
    add_chunks(doc_id, chunk_records, db_path=db_path)

    faiss_store.save()

    return {
        "doc_id": doc_id,
        "name": file_name,
        "chunks_indexed": len(chunks),
    }
