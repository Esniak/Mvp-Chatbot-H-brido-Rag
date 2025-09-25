# Repository Guidelines

## Project Structure & Module Organization
- `src/api/` expone la API FastAPI; `main.py` define `/ask` y la lógica de conversación.
- `src/rag/` contiene recuperación y embeddings; `retriever.py` gestiona FAISS y `src/ingest/` prepara el índice.
- `src/common/` aloja utilidades compartidas como el registrador SQLite.
- `src/panel/` incluye la app Streamlit de métricas; los scripts operativos residen en `scripts/`.
- Datos estáticos viven en `data/` (FAQ CSV, índices, `logs.db`); `prompts/` guarda mensajes del sistema.

## Build, Test, and Development Commands
- `bash scripts/run_ingest_online.sh` genera índice FAISS desde el CSV oficial.
- `bash scripts/run_api_local.sh` o `bash scripts/run_api_local_8001.sh` sirven la API en `127.0.0.1` (puerto 8000/8001).
- `bash scripts/run_panel.sh` levanta el panel Streamlit en `http://127.0.0.1:8501`.
- `http --file scripts/test_ask.http` o `curl` equivalente validan `/ask` rápidamente.

## Coding Style & Naming Conventions
- Python 3.11, indentación de 4 espacios, tipado estático con `typing` y `pydantic` para contratos.
- Evita reformateos masivos: sigue el estilo existente, comentarios breves solo cuando aporten contexto.
- Nombra funciones y módulos en snake_case; clases en PascalCase; JSON expuesto en español (ej. `respuesta`, `fuentes`).

## Testing Guidelines
- No hay suite formal; valida `/ask` con `curl` (sin y con `show_sources=true`) y verifica registros en `data/logs.db`.
- Para cambios RAG, reejecuta `run_ingest_online.sh` y comprueba la recuperación offline (`OFFLINE=1`).
- Antes de fusionar, revisa que el panel Streamlit muestra al menos un turno de ejemplo sin errores.

## Commit & Pull Request Guidelines
- Mensajes de commit en imperativo corto (`Actualiza panel`, `Corrige logging`); agrupa cambios relacionados.
- Las PR deben describir propósito, pasos de verificación manual (cURL/panel) y mencionar scripts ejecutados.
- Incluye capturas o salidas relevantes solo cuando aporten claridad; enlaza issue/jira si aplica.

## Security & Configuration Tips
- Nunca subas claves: usa `.env` local y variables `OPENAI_API_KEY`, `OPENAI_MODEL`, `LOG_DB_PATH`.
- Respeta los modos `OFFLINE` para evitar llamadas externas en entornos restringidos.
- Verifica permisos de `data/` antes de demostrar la app; limpia `logs.db` si contiene PII accidental.
