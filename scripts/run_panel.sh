#!/usr/bin/env bash
set -euo pipefail
exec .venv/bin/streamlit run src/panel/app.py --server.port 8501

