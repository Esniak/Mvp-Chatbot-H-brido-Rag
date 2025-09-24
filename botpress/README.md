# Botpress → Kaabil RAG API

1. Crea un flujo con un **Node de Captura** (texto libre).
2. Añade un **Call API** apuntando a `POST http://TU_HOST:8000/ask` con body `{ "query": "{{state.user_query}}" }`.
3. Enruta la **respuesta** a un **Send Message** con `response.answer`.
4. Controla fallback: si `used_evidence == false`, ofrece transferencia a humano.
