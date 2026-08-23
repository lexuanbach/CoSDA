#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scenarios/aws_g6e4xlarge_vllm_revised.yaml}"
RUN_NAME="${RUN_NAME:-aws_vllm_revised_full_20260520}"
OUT_ROOT="${OUT_ROOT:-/mnt/cosda/runs/${RUN_NAME}_hf_paircheck}"
GPU_ID="${GPU_ID:-auto}"
WAIT_FOR_FREE_GPU="${WAIT_FOR_FREE_GPU:-1}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
MODEL="${MODEL:-FacebookAI/xlm-roberta-base}"
SEED="${SEED:-13}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
COSDA_HF_DETERMINISTIC="${COSDA_HF_DETERMINISTIC:-1}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"

export OUT_ROOT OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM COSDA_HF_DETERMINISTIC CUDA_LAUNCH_BLOCKING

if [ "$GPU_ID" = "auto" ] || [ "$WAIT_FOR_FREE_GPU" = "1" ]; then
  GPU_ID="$(python - "$GPU_ID" "$MAX_GPU_MEMORY_USED_MIB" "$MAX_GPU_UTILIZATION" "$GPU_POLL_SECONDS" <<'PY'
from __future__ import annotations

import subprocess
import sys
import time

target = sys.argv[1]
max_mem = int(sys.argv[2])
max_util = int(sys.argv[3])
poll = float(sys.argv[4])


def stats() -> dict[str, tuple[int, int]]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "nvidia-smi failed")
    out = {}
    for line in proc.stdout.splitlines():
        idx, mem, util = [part.strip() for part in line.split(",")[:3]]
        out[idx] = (int(mem), int(util))
    return out


while True:
    current = stats()
    candidates = sorted(current) if target == "auto" else [target]
    for gpu in candidates:
        mem, util = current.get(gpu, (10**9, 100))
        if mem <= max_mem and util <= max_util:
            print(gpu)
            raise SystemExit(0)
    detail = " | ".join(f"{gpu}:mem={mem}MiB,util={util}%" for gpu, (mem, util) in sorted(current.items()))
    print(f"waiting for free GPU under thresholds: {detail}", file=sys.stderr, flush=True)
    time.sleep(poll)
PY
)"
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"

if [ -f scratch.env ]; then
  # shellcheck disable=SC1091
  source scratch.env
fi

COSDA="${COSDA:-$(command -v cosda || true)}"
if [ -z "$COSDA" ] && [ -x ".venv/bin/cosda" ]; then
  COSDA=".venv/bin/cosda"
fi
if [ -z "$COSDA" ]; then
  echo "cosda command not found; activate conda/venv or set COSDA=/path/to/cosda" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

run_pair() {
  local language="$1"
  local baseline="$2"
  local base="/mnt/cosda/runs/${RUN_NAME}/news_topic_classification/${language}/b64_m3_s${SEED}"
  local selected="${base}/selected/${baseline}.jsonl"
  local out_dir="${OUT_ROOT}/news_topic_classification/${language}/${baseline}"
  echo "== deterministic paircheck language=${language} baseline=${baseline} =="
  "$COSDA" train-hf \
    --manifest data/manifest/datasets.json \
    --dataset-id masakhane/masakhanews \
    --language "$language" \
    --gold "${base}/gold.jsonl" \
    --selected "$selected" \
    --out-dir "$out_dir" \
    --model "$MODEL" \
    --epochs 3 \
    --batch-size 16 \
    --max-length 256 \
    --seed "$SEED" \
    --deterministic
}

run_pair amh naive
run_pair amh cosda_equal_budget
run_pair swa naive
run_pair swa cosda_equal_budget

python - <<'PY'
from __future__ import annotations

import json
import os
from pathlib import Path

root = Path(os.environ["OUT_ROOT"])
rows = []
for language in ["amh", "swa"]:
    values = {}
    for baseline in ["naive", "cosda_equal_budget"]:
        path = root / "news_topic_classification" / language / baseline / "test_metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        value = metrics.get("test_macro_f1") or metrics.get("macro_f1") or metrics.get("eval_macro_f1")
        values[baseline] = float(value)
    rows.append(
        {
            "language": language,
            "naive": values["naive"],
            "cosda_equal_budget": values["cosda_equal_budget"],
            "delta": values["cosda_equal_budget"] - values["naive"],
        }
    )
out = root / "paircheck_summary.json"
out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
print(out)
print(json.dumps({"rows": rows}, indent=2))
PY
