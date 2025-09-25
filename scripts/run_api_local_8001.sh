#!/usr/bin/env bash
set -euo pipefail
export OFFLINE=0
.venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
