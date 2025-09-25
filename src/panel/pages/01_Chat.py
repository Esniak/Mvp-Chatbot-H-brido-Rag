from __future__ import annotations

import os
import uuid
import requests
import streamlit as st

st.set_page_config(page_title="Chat demo — Kaabil RAG", page_icon="💬", layout="wide")

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

st.title("💬 Chat demo — Kaabil RAG")
st.caption(f"API: {API_BASE}/ask  ·  Session: {st.session_state.session_id}")

with st.form("ask_form", clear_on_submit=False):
    query = st.text_input("Escribe tu pregunta", placeholder="Ej.: ¿Cuál es vuestro horario de atención?")
    show_sources = st.checkbox("Mostrar fuentes", value=False)
    submitted = st.form_submit_button("Enviar")

if submitted and query.strip():
    payload = {"query": query.strip(), "show_sources": bool(show_sources)}
    headers = {
        "Content-Type": "application/json",
        "X-Session-Id": st.session_state.session_id,
        "User-Agent": "Kaabil-Panel/Chat",
    }
    try:
        resp = requests.post(f"{API_BASE}/ask", json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        st.error(f"No se pudo conectar a la API en {API_BASE}/ask. ¿Está levantada? Detalle: {e}")
    else:
        respuesta = (data or {}).get("respuesta", "").strip()
        if respuesta:
            st.success(respuesta)
        else:
            st.warning("La API respondió sin texto.")

        if show_sources:
            fuentes = (data or {}).get("fuentes") or []
            evidencia = bool((data or {}).get("evidencia"))
            if fuentes:
                st.divider()
                st.caption("Fuentes")
                for c in fuentes:
                    st.write(f"• {c}")
            st.caption(f"Evidencia: {'sí' if evidencia else 'no'}")

st.divider()
st.caption("Consejo: si cambias el puerto/host de la API, exporta API_BASE_URL antes de abrir el panel.")
st.code('export API_BASE_URL="http://127.0.0.1:8001"', language="bash")
