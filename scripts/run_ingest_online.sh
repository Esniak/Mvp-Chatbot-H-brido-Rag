#!/usr/bin/env bash
set -euo pipefail

export OFFLINE=0

python -m pip install -r requirements.txt

python src/ingest/ingest.py --csv data/faqs/faqs_es.csv

python - <<'PY'
import faiss, json
index = faiss.read_index('data/index.faiss')
with open('data/index_meta.json', 'r', encoding='utf-8') as meta_file:
    meta = json.load(meta_file)
print(f"ntotal={index.ntotal} items={len(meta['items'])}")
PY
