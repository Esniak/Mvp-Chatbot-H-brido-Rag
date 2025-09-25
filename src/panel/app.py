from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_DB = os.getenv("LOG_DB_PATH")

if ENV_DB:
    candidate = Path(ENV_DB).expanduser()
    DB_PATH = candidate if candidate.is_absolute() else (BASE_DIR / candidate)
else:
    DB_PATH = BASE_DIR / "data" / "logs.db"
DB_PATH = DB_PATH.resolve()


st.set_page_config(page_title="Kaabil RAG - Panel", layout="wide")
st.title("Panel de métricas - Kaabil RAG")


@st.cache_resource(show_spinner=False)
def _connect(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


connection = _connect(DB_PATH)

if connection is None:
    st.info("Aún no existe el archivo de logs. Genera interacciones para verlo aquí.")
    st.stop()


def _parse_citations(raw: str | None) -> int:
    if not raw:
        return 0
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return 0
    if isinstance(parsed, (list, tuple, set)):
        return len(parsed)
    return 0


def _iso_with_z(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat() + "Z"


@st.cache_data(show_spinner=False)
def load_data(start: datetime, end: datetime) -> pd.DataFrame:
    query = """
        SELECT
          ts,
          session_id,
          ip,
          user_agent,
          query,
          answer,
          used_evidence,
          citations,
          latency_ms,
          provider,
          model,
          topk,
          threshold
        FROM turns
        WHERE ts BETWEEN ? AND ?
    """
    frame = pd.read_sql_query(
        query,
        connection,
        params=(_iso_with_z(start), _iso_with_z(end)),
    )
    if frame.empty:
        return frame

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"])
    frame["num_citations"] = frame["citations"].apply(_parse_citations)
    frame["used_evidence"] = frame["used_evidence"].fillna(0).astype(int)
    return frame


with st.sidebar:
    st.header("Filtros")
    today = datetime.utcnow().date()
    default_start = today - timedelta(days=7)
    date_range = st.date_input(
        "Rango de fechas",
        value=(default_start, today),
        max_value=today,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = default_start
        end_date = today

    text_filter = st.text_input("Buscar en consulta (contiene)").strip()

    session_filter = "Todos"
    model_filter = "Todos"

    st.divider()
    st.caption("Opciones avanzadas")
    show_pii = st.toggle("Mostrar IP/UA (avanzado)", value=False)


start_dt = datetime.combine(start_date, datetime.min.time())
end_dt = datetime.combine(end_date, datetime.max.time())

data = load_data(start_dt, end_dt)

if data.empty:
    st.info("No hay registros para el rango seleccionado.")
    st.stop()


session_options = sorted(data["session_id"].dropna().unique().tolist())
session_filter = st.sidebar.selectbox("Session ID", ["Todos"] + session_options)

model_options = sorted(data["model"].dropna().unique().tolist())
model_filter = st.sidebar.selectbox("Modelo", ["Todos"] + model_options)


filtered = data.copy()
if text_filter:
    filtered = filtered[filtered["query"].str.contains(text_filter, case=False, na=False)]

if session_filter != "Todos":
    filtered = filtered[filtered["session_id"] == session_filter]

if model_filter != "Todos":
    filtered = filtered[filtered["model"] == model_filter]


if filtered.empty:
    st.warning("No hay coincidencias con los filtros aplicados.")
    st.stop()


total_turns = int(filtered.shape[0])
unique_sessions = int(filtered["session_id"].nunique())
avg_latency = round(filtered["latency_ms"].dropna().mean(), 1) if "latency_ms" in filtered else 0
evidence_rate = (
    round((filtered["used_evidence"].sum() / total_turns) * 100, 1) if total_turns else 0
)


cols = st.columns(4)
cols[0].metric("Total de turnos", f"{total_turns}")
cols[1].metric("Sesiones únicas", f"{unique_sessions}")
cols[2].metric("Latencia media (ms)", f"{avg_latency}")
cols[3].metric("% con evidencia", f"{evidence_rate}%")


daily_counts = (
    filtered.sort_values("ts")
    .set_index("ts")
    .resample("1H" if (end_dt - start_dt).days <= 2 else "1D")
    .size()
)
daily_counts = daily_counts.rename("turnos").reset_index()

st.subheader("Actividad en el tiempo")
st.line_chart(daily_counts, x="ts", y="turnos")

st.subheader("Distribución de latencia (ms)")
latency_series = filtered["latency_ms"].dropna()
if latency_series.empty:
    st.caption("Sin datos de latencia disponibles.")
else:
    bins = min(20, max(5, latency_series.nunique()))
    hist = (
        pd.cut(latency_series, bins=bins, include_lowest=True)
        .value_counts()
        .sort_index()
        .rename_axis("rango")
        .reset_index(name="turnos")
    )
    hist["rango"] = hist["rango"].astype(str)
    st.bar_chart(hist.set_index("rango"))


st.subheader("Detalles de turnos")
table = filtered.sort_values("ts", ascending=False).head(200).copy()
table["respuesta"] = table["answer"].fillna("").str.slice(0, 120)
table["used_evidence"] = table["used_evidence"].astype(bool)
columns_to_show = [
    "ts",
    "session_id",
    "query",
    "respuesta",
    "latency_ms",
    "model",
    "used_evidence",
    "num_citations",
]
if show_pii:
    columns_to_show.extend(["ip", "user_agent"])

st.dataframe(table[columns_to_show], width="stretch")
