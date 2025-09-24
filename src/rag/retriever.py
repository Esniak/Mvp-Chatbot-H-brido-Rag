"""FAISS-based retriever utilities for the RAG pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import faiss
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

EMBED_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TOPK = int(os.getenv("RETRIEVAL_TOPK", 4))
REQUEST_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "60"))


def _build_headers() -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY no está configurada en el entorno")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def get_embedding(text: str) -> np.ndarray:
    payload = {"input": text, "model": EMBED_MODEL}
    headers = _build_headers()

    try:
        response = requests.post(
            EMBED_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        message = getattr(exc.response, "text", str(exc))
        raise RuntimeError(f"No se pudo generar el embedding: {message}") from exc

    data = response.json()
    try:
        embedding = data["data"][0]["embedding"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError("Respuesta inesperada del endpoint de embeddings") from exc
    return np.array(embedding, dtype="float32")


class Retriever:
    """Simple wrapper around a FAISS index with stored metadata."""

    def __init__(self, index_path: str, meta_path: str):
        index_file = Path(index_path)
        meta_file = Path(meta_path)
        if not index_file.exists() or not meta_file.exists():
            missing = ", ".join(str(p) for p in (index_file, meta_file) if not p.exists())
            raise FileNotFoundError(
                "No se encontraron los artefactos del índice. Ejecuta la ingesta "
                f"({missing})."
            )

        try:
            self.index = faiss.read_index(str(index_file))
        except RuntimeError as exc:
            raise RuntimeError("No se pudo cargar el índice FAISS") from exc

        with meta_file.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        items = metadata.get("items")
        if not isinstance(items, list):
            raise ValueError("El archivo de metadatos no contiene una lista de 'items'")

        self.meta: Sequence[dict] = items

    def search(self, query: str) -> List[Tuple[dict, float]]:
        query_embedding = get_embedding(query)
        norm = np.linalg.norm(query_embedding)
        if not norm:
            raise ValueError("El embedding calculado tiene norma cero")

        query_embedding = query_embedding / norm
        distances, indices = self.index.search(np.expand_dims(query_embedding, 0), TOPK)
        scores = distances[0].tolist()
        idxs = indices[0].tolist()

        documents = []
        for idx in idxs:
            if 0 <= idx < len(self.meta):
                documents.append(self.meta[idx])
            else:  # pragma: no cover - índices corruptos
                documents.append({})
        return list(zip(documents, scores))
