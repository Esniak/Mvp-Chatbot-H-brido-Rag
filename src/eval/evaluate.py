"""Offline evaluation helper for the RAG retriever."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rag.retriever import Retriever  # noqa: E402 (path adjustment above)


def load_eval_set(path: Path) -> Sequence[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("El archivo de evaluación debe contener una lista de casos")
    return data


def has_required_tokens(answer: str, tokens: Iterable[str]) -> bool:
    text = answer.lower()
    return all(token.lower() in text for token in tokens)


def evaluate(index_path: Path, meta_path: Path, eval_path: Path, threshold: float) -> dict:
    if not index_path.exists() or not meta_path.exists():
        missing = ", ".join(str(p) for p in (index_path, meta_path) if not p.exists())
        raise FileNotFoundError(
            "No se encontró el índice FAISS. Ejecuta la ingesta antes de evaluar "
            f"({missing})."
        )

    retriever = Retriever(str(index_path), str(meta_path))
    eval_set = load_eval_set(eval_path)

    hits = 0
    no_evidence = 0

    for item in eval_set:
        query = item.get("query")
        if not query:
            continue

        results = retriever.search(query)
        passing = [(doc, score) for doc, score in results if score >= threshold]
        if not passing:
            no_evidence += 1
            continue

        answers = "\n".join(doc["answer"] for doc, _ in passing)
        if has_required_tokens(answers, item.get("must_contain", [])):
            hits += 1

    total = len(eval_set)
    hit_rate = hits / total if total else 0.0

    return {
        "total": total,
        "hits": hits,
        "no_evidence": no_evidence,
        "hit_rate": round(hit_rate, 3),
    }


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evalúa el índice RAG con un conjunto JSON")
    parser.add_argument("--index", default="data/index.faiss", help="Ruta al índice FAISS")
    parser.add_argument(
        "--meta", default="data/index_meta.json", help="Ruta al archivo de metadatos"
    )
    parser.add_argument(
        "--eval", default="evaluation/eval_set.json", help="Ruta al conjunto de evaluación"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("SCORE_THRESHOLD", 0.25)),
        help="Umbral mínimo de score para contar evidencia",
    )
    args = parser.parse_args()

    index_path = Path(args.index)
    meta_path = Path(args.meta)
    eval_path = Path(args.eval)

    results = evaluate(index_path, meta_path, eval_path, args.threshold)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
