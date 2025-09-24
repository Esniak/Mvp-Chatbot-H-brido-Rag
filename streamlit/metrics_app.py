import json

import streamlit as st

st.title("Métricas RAG (MVP)")

st.markdown("Sube el resultado de evaluate.py (JSON) para visualizar métricas.")

uploaded = st.file_uploader("Resultado JSON", type=["json"])
if uploaded:
    data = json.load(uploaded)
    st.metric("Total preguntas", data.get("total", 0))
    st.metric("Hit rate", data.get("hit_rate", 0))
    st.metric("Sin evidencia", data.get("no_evidence", 0))
