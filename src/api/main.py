"""FastAPI application exposing the RAG chatbot service."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.rag.retriever import Retriever
from src.common.logs import init_db, log_turn

load_dotenv()

API = FastAPI(title="Kaabil RAG API")
app = API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", 0.30))
INDEX_PATH = Path(os.getenv("INDEX_PATH", "data/index.faiss"))
META_PATH = Path(os.getenv("META_PATH", "data/index_meta.json"))
RETRIEVAL_TOPK = int(os.getenv("RETRIEVAL_TOPK", "4"))

init_db()

with Path("prompts/system.txt").open("r", encoding="utf-8") as file:
    SYSTEM_PROMPT = file.read()

_retriever: Retriever | None = None


def _is_offline() -> bool:
    return os.getenv("OFFLINE", "0") == "1"


def _normalize(text: str) -> list[str]:
    tokens = []
    for token in text.lower().split():
        simple = (
            token.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        if token.isalpha() or simple.isalpha():
            tokens.append(simple)
    return tokens


def _token_overlap(query: str, document: str) -> float:
    qs = set(_normalize(query))
    ds = set(_normalize(document))
    if not qs:
        return 0.0
    return len(qs & ds) / max(1, len(qs))


def _select_relevant(
    evidence: list[tuple[dict, float]], query: str, k: int = 2
) -> list[tuple[dict, float]]:
    qn = query.strip().lower()
    exact = [
        (doc, score)
        for doc, score in evidence
        if qn
        and (
            qn in (doc.get("question", "").strip().lower())
            or (doc.get("question", "").strip().lower()) in qn
        )
    ]
    if exact:
        return exact[:1]

    scored = [
        (doc, _token_overlap(query, doc.get("question", "")))
        for doc, _ in evidence
    ]
    scored = [(doc, overlap) for doc, overlap in scored if overlap >= 0.5]
    scored.sort(key=lambda item: item[1], reverse=True)
    top = [(doc, 1.0) for doc, _ in scored[:k]]
    return top or evidence[:1]


def _format_citation(doc: dict) -> str:
    cat = doc.get("category") or doc.get("Category") or "Fuente"
    question = doc.get("question") or doc.get("Question") or "FAQ"
    url = doc.get("source_url") or doc.get("url") or ""
    citation = f"{cat} – {question}".strip()
    if url:
        citation = f"{citation} ({url})"
    return citation


def _build_context(evidence: list[tuple[dict, float]]) -> str:
    sections: list[str] = []
    for doc, _ in evidence:
        category = (doc.get("category") or doc.get("Category") or "Información").strip()
        question = (doc.get("question") or doc.get("Question") or "").strip()
        answer = (doc.get("answer") or doc.get("Answer") or "").strip()

        lines = [f"Categoría: {category}"]
        if question:
            lines.append(f"Pregunta relacionada: {question}")
        if answer:
            lines.append(f"Respuesta sugerida: {answer}")

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def _clean_answer(text: str) -> str:
    if not text:
        return ""

    cleaned = re.sub(r"\[CAT:[^\]]*\]", "", text)
    cleaned = re.sub(r"(?im)^\s*(Fuentes?:|Sources?:).*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*(?:Q|A)\s*:\s*", "", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(stripped)

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


class AskIn(BaseModel):
    query: str
    show_sources: bool = False


class AskOut(BaseModel):
    respuesta: str
    fuentes: List[str] | None = None
    evidencia: bool | None = None


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
    nano = "gpt-5-nano" in OPENAI_MODEL.lower()
    payload = {"model": OPENAI_MODEL, "messages": messages}
    if not nano:
        payload["temperature"] = 0.0
    else:
        for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            payload.pop(key, None)

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


@API.post("/ask", response_model=AskOut, response_model_exclude_none=True)
def ask(payload: AskIn, request: Request) -> AskOut:
    start = time.perf_counter()
    session_id = request.headers.get("X-Session-Id") or uuid.uuid4().hex
    try:
        retriever = _load_retriever()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    offline_mode = _is_offline()
    items: List[Tuple[dict, float]] = retriever.search(payload.query)
    evidence = [(doc, score) for doc, score in items if score >= SCORE_THRESHOLD]
    if offline_mode and not evidence:
        evidence = items
    evidence = _select_relevant(evidence, payload.query, k=2)

    if not evidence:
        fallback = (
            "No encuentro información fiable para responder a esto. "
            "¿Quieres que te ponga con una persona del equipo?"
        )
        result: dict[str, object] = {"respuesta": _clean_answer(fallback)}
        if payload.show_sources:
            result["fuentes"] = []
            result["evidencia"] = False
        ask_out = AskOut(**result)
        latency_ms = int((time.perf_counter() - start) * 1000)
        try:
            log_turn(
                ts=datetime.utcnow().isoformat(timespec="seconds") + "Z",
                session_id=session_id,
                ip=(request.client.host if request and request.client else None),
                user_agent=request.headers.get("User-Agent"),
                query=payload.query,
                answer=ask_out.respuesta,
                used_evidence=1 if (hasattr(ask_out, "evidencia") and ask_out.evidencia) else 0,
                citations=json.dumps(getattr(ask_out, "fuentes", []) or [], ensure_ascii=False),
                latency_ms=latency_ms,
                provider="openai",
                model=OPENAI_MODEL,
                topk=RETRIEVAL_TOPK,
                threshold=SCORE_THRESHOLD,
            )
        except Exception:
            pass
        return ask_out

    context = _build_context(evidence)
    citations: List[str] = []
    for doc, _ in evidence:
        citation = _format_citation(doc)
        if citation and citation not in citations:
            citations.append(citation)

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

    if offline_mode:
        exact_doc = next(
            (
                doc
                for doc, _ in evidence
                if payload.query.strip().lower()
                and (
                    payload.query.strip().lower()
                    in doc.get("question", "").strip().lower()
                    or doc.get("question", "").strip().lower()
                    in payload.query.strip().lower()
                )
            ),
            None,
        )
        chosen = exact_doc or (evidence[0][0] if evidence else {})
        answer = (chosen or {}).get("answer", "")
    else:
        answer = _call_openai(messages)

    clean_answer = _clean_answer(answer)
    result: dict[str, object] = {"respuesta": clean_answer}
    if payload.show_sources:
        result["fuentes"] = citations
        result["evidencia"] = bool(evidence)

    ask_out = AskOut(**result)
    latency_ms = int((time.perf_counter() - start) * 1000)
    try:
        log_turn(
            ts=datetime.utcnow().isoformat(timespec="seconds") + "Z",
            session_id=session_id,
            ip=(request.client.host if request and request.client else None),
            user_agent=request.headers.get("User-Agent"),
            query=payload.query,
            answer=ask_out.respuesta,
            used_evidence=1 if (hasattr(ask_out, "evidencia") and ask_out.evidencia) else 0,
            citations=json.dumps(getattr(ask_out, "fuentes", []) or [], ensure_ascii=False),
            latency_ms=latency_ms,
            provider="openai",
            model=OPENAI_MODEL,
            topk=RETRIEVAL_TOPK,
            threshold=SCORE_THRESHOLD,
        )
    except Exception:
        pass

    return ask_out
