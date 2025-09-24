# Chatbot Híbrido con RAG (MVP)

Este MVP expone un microservicio RAG (FastAPI) para responder con **citas** a partir de un dataset de **FAQs**. Incluye:

- **Ingesta** de datos y construcción de índice **FAISS** con embeddings.
- **API** `/ask` para recuperar evidencias + generar respuesta con política "**solo con evidencia**".
- **Evaluación offline** con `evaluation/eval_set.json`.
- **Panel** Streamlit de métricas básicas.

> **Nota**: piensa en este repo como "núcleo RAG". La conexión a **Web** o **WhatsApp/Botpress** se hace apuntando a la API `/ask`.

## Requisitos
- Python 3.10+
- Dependencias: `pip install -r requirements.txt`
- Variables en `.env` (usa `.env.sample`):
  - `OPENAI_API_KEY=`
  - `OPENAI_MODEL=` (ej: `gpt-4o-mini` o equivalente disponible)
  - `EMBEDDING_MODEL=` (ej: `text-embedding-3-small`)
  - `RETRIEVAL_TOPK=4`
  - `SCORE_THRESHOLD=0.25`  # umbral mínimo para considerar evidencia útil

## Uso rápido
1. **Ingesta**: `python src/ingest/ingest.py --csv data/faqs/faqs_es.csv`  → genera `data/index.faiss` y `data/index_meta.json`.
2. **Levantar API**: `uvicorn src.api.main:app --reload` (por defecto en `http://127.0.0.1:8000`).
3. **Probar**: `POST /ask` con `{ "query": "¿Cuál es el horario de atención?" }`.
4. **Evaluación**: `python src/eval/evaluate.py` (lee `evaluation/eval_set.json`).
5. **Panel**: `streamlit run streamlit/metrics_app.py`.

## Buenas prácticas
- Mantén el **dominio acotado** (FAQs + docs soporte), y evita mezclar temas no relacionados.
- Actualiza `policies/policies.md` si cambian los requisitos (GDPR/AI Act, disclaimers y retención de logs).
- Versiona `prompts/system.txt` y `eval_set.json` para trazabilidad.
