"""FastAPI application exposing the RAG chatbot service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.retriever import Retriever

load_dotenv()

API = FastAPI(title="Kaabil RAG API")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.25))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/index.faiss"))
META_PATH = Path(os.getenv("META_PATH", "data/index_meta.json"))

with Path("prompts/system.txt").open("r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

_retriever: Retriever | None = None


class AskIn(BaseModel):
    query: str


class AskOut(BaseModel):
    answer: str
    citations: List[str]
    used_evidence: bool


def _load_retriever() -> Retriever:
    """Initialise (once) the FAISS retriever backing the API."""

    global _retriever
    if _retriever is not None:
        return _retriever

    if not INDEX_PATH.exists() or not META_PATH.exists():
        missing = ", ".join(
            [str(path) for path in (INDEX_PATH, META_PATH) if not path.exists()]
        )
        raise RuntimeError(
            "El índice FAISS no está disponible. Ejecuta el script de ingesta "
            f"para generarlo ({missing})."
        )

    _retriever = Retriever(str(INDEX_PATH), str(META_PATH))
    return _retriever


def _format_citation(doc: dict) -> str:
    source = doc.get("source_url")
    base = f"{doc['category']} – {doc['question']}"
    return f"{base} ({source})" if source else base


def _call_openai(messages: List[dict]) -> str:
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY no está configurada en el entorno",
        )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": OPENAI_MODEL, "messages": messages, "temperature": 0.0}

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        message = getattr(exc.response, "text", str(exc))
        raise HTTPException(status_code=502, detail=f"Fallo al consultar OpenAI: {message}")

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="Respuesta inesperada de OpenAI") from exc


@API.post("/ask", response_model=AskOut)
def ask(payload: AskIn) -> AskOut:
    try:
        retriever = _load_retriever()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    items: List[Tuple[dict, float]] = retriever.search(payload.query)
    evidence = [(doc, score) for doc, score in items if score >= SCORE_THRESHOLD]

    if not evidence:
        return AskOut(
            answer=(
                "No encuentro información fiable para responder a esto. "
                "¿Quieres que te ponga con una persona del equipo?"
            ),
            citations=[],
            used_evidence=False,
        )

    context_lines = [
        f"[CAT:{doc['category']}] Q:{doc['question']}\nA:{doc['answer']}"
        for doc, _ in evidence
    ]
    citations = [_format_citation(doc) for doc, _ in evidence]
    context = "\n\n".join(context_lines)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Contexto (evidencia):\n{context}\n\n"
                f"Pregunta del usuario: {payload.query}"
            ),
        },
    ]

    answer = _call_openai(messages)
    if citations:
        answer = f"{answer}\n\nFuentes: {', '.join(citations)}"

    return AskOut(answer=answer, citations=citations, used_evidence=True)
