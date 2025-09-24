#!/usr/bin/env bash
set -euo pipefail

export OFFLINE=0

uvicorn src.api.main:app --host 127.0.0.1 --port 8000
