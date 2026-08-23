#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-$ROOT/.venv/bin/python}"
COSDA="${COSDA:-$ROOT/.venv/bin/cosda}"

"$PY" scripts/prepare_raw_data.py
"$PY" scripts/build_manifest.py

BASE="runs/smoke/sentiment_classification/swa/b8_m1_s13"
"$COSDA" make-seeds \
  --manifest data/manifest/datasets.json \
  --dataset-id masakhane/afrisenti \
  --language swa \
  --budget 8 \
  --seed 13 \
  --out "$BASE/gold.jsonl" \
  --per-cell

"$COSDA" generate \
  --config configs/scenarios/smoke.yaml \
  --seed-file "$BASE/gold.jsonl" \
  --out "$BASE/candidates.jsonl" \
  --candidates-per-seed 1

"$COSDA" counterfactuals \
  --config configs/scenarios/smoke.yaml \
  --candidates "$BASE/candidates.jsonl" \
  --out "$BASE/candidates_cf.jsonl"

"$COSDA" judge \
  --config configs/scenarios/smoke.yaml \
  --candidates "$BASE/candidates_cf.jsonl" \
  --out "$BASE/candidates_judged.jsonl"

"$COSDA" audit \
  --config configs/scenarios/smoke.yaml \
  --manifest data/manifest/datasets.json \
  --dataset-id masakhane/afrisenti \
  --language swa \
  --budget 8 \
  --seed 13 \
  --gold "$BASE/gold.jsonl" \
  --candidates "$BASE/candidates_judged.jsonl" \
  --out "$BASE/audit.jsonl"

"$COSDA" select \
  --config configs/scenarios/smoke.yaml \
  --audit "$BASE/audit.jsonl" \
  --gold "$BASE/gold.jsonl" \
  --out "$BASE/selected/cosda.jsonl" \
  --budget 8 \
  --baseline cosda

"$COSDA" evaluate \
  --manifest data/manifest/datasets.json \
  --dataset-id masakhane/afrisenti \
  --language swa \
  --gold "$BASE/gold.jsonl" \
  --selected "$BASE/selected/cosda.jsonl" \
  --out "$BASE/results/cosda_lightweight.json"

cat "$BASE/results/cosda_lightweight.json"
