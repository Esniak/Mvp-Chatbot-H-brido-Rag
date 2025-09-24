"""Utility script to build the FAISS index from a FAQs CSV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import faiss
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

EMBED_URL = "https://api.openai.com/v1/embeddings"
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
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


def embed_texts(texts: Iterable[str]) -> List[np.ndarray]:
    vectors = []
    for text in texts:
        vectors.append(get_embedding(text))
    if not vectors:
        raise ValueError("El CSV no contiene preguntas/respuestas para indexar")
    return vectors


def build_index(vectors: List[np.ndarray]) -> faiss.Index:
    dim = len(vectors[0])
    index = faiss.IndexFlatIP(dim)
    matrix = np.vstack(vectors)
    faiss.normalize_L2(matrix)
    index.add(matrix)
    return index


def save_metadata(df: pd.DataFrame, path: Path) -> None:
    items = []
    for i, row in df.iterrows():
        items.append(
            {
                "id": int(i),
                "category": row.get("category", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "source_url": row.get("source_url", ""),
            }
        )

    metadata = {"items": items, "embedding_model": EMBED_MODEL}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el índice FAISS para el chatbot")
    parser.add_argument("--csv", required=True, help="Ruta al CSV con FAQs")
    parser.add_argument("--out_index", default="data/index.faiss", help="Destino del índice")
    parser.add_argument("--out_meta", default="data/index_meta.json", help="Destino de metadatos")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV en {csv_path}")

    df = pd.read_csv(csv_path)
    texts = (df["question"].fillna("") + "\n" + df["answer"].fillna("")).tolist()
    vectors = embed_texts(texts)

    index = build_index(vectors)

    out_index = Path(args.out_index)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_index))

    save_metadata(df, Path(args.out_meta))

    print(f"Índice guardado en {out_index} y metadatos en {args.out_meta}")


if __name__ == "__main__":
    main()
