#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scenarios/aws_g6e4xlarge_vllm.yaml}"
RUN_NAME="${RUN_NAME:-aws_vllm_pilot}"
COSDA_RESOURCE_PROFILE="${COSDA_RESOURCE_PROFILE:-balanced}"
SCENARIO_JOBS="${SCENARIO_JOBS:-}"
JUDGE_JOBS="${JUDGE_JOBS:-}"
TRAIN_JOBS="${TRAIN_JOBS:-}"
GPU_IDS="${GPU_IDS:-auto}"
GPU_POLICY="${GPU_POLICY:-wait-free}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
VLLM_SESSION="${VLLM_SESSION:-cosda_vllm_${RUN_NAME}}"
ALLOW_REPLACE_VLLM_SESSION="${ALLOW_REPLACE_VLLM_SESSION:-0}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-0}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.86}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---generation-config vllm}"
GENERATION_RPS="${GENERATION_RPS:-1.0}"
COUNTERFACTUAL_RPS="${COUNTERFACTUAL_RPS:-1.0}"
JUDGE_RPS="${JUDGE_RPS:-0.5}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

case "$COSDA_RESOURCE_PROFILE" in
  smoke)
    SCENARIO_JOBS="${SCENARIO_JOBS:-1}"
    JUDGE_JOBS="${JUDGE_JOBS:-1}"
    TRAIN_JOBS="${TRAIN_JOBS:-1}"
    ;;
  balanced)
    SCENARIO_JOBS="${SCENARIO_JOBS:-2}"
    JUDGE_JOBS="${JUDGE_JOBS:-2}"
    TRAIN_JOBS="${TRAIN_JOBS:-4}"
    ;;
  conservative)
    SCENARIO_JOBS="${SCENARIO_JOBS:-1}"
    JUDGE_JOBS="${JUDGE_JOBS:-1}"
    TRAIN_JOBS="${TRAIN_JOBS:-2}"
    ;;
  aggressive)
    SCENARIO_JOBS="${SCENARIO_JOBS:-3}"
    JUDGE_JOBS="${JUDGE_JOBS:-2}"
    TRAIN_JOBS="${TRAIN_JOBS:-4}"
    ;;
  *)
    echo "Unknown COSDA_RESOURCE_PROFILE=$COSDA_RESOURCE_PROFILE; use smoke, conservative, balanced, or aggressive" >&2
    exit 1
    ;;
esac
export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM

if [ -f scratch.env ]; then
  # shellcheck disable=SC1091
  source scratch.env
fi

RUNS_DIR="${COSDA_RUNS:-/mnt/cosda/runs}"
ARTIFACT_DIR="${COSDA_ARTIFACTS:-/mnt/cosda/artifacts}"
LOG_DIR="${COSDA_LOGS:-/mnt/cosda/logs}"
mkdir -p "$RUNS_DIR" "$ARTIFACT_DIR" "$LOG_DIR"

COSDA="${COSDA:-$(command -v cosda || true)}"
if [ -z "$COSDA" ] && [ -x ".venv/bin/cosda" ]; then
  COSDA=".venv/bin/cosda"
fi
if [ -z "$COSDA" ]; then
  echo "cosda command not found; activate conda/venv or set COSDA=/path/to/cosda" >&2
  exit 1
fi

config_value() {
  python - "$CONFIG" "$1" <<'PY'
from pathlib import Path
import sys
import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text()) or {}
value = cfg
for part in sys.argv[2].split("."):
    value = value[part]
print(value)
PY
}

GEN_MODEL="${GEN_MODEL:-$(config_value generation.model)}"
JUDGE_MODEL="${JUDGE_MODEL:-$(config_value judge.model)}"

wait_for_vllm() {
  local expected_model="$1"
  local models_url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
  for _ in $(seq 1 120); do
    if curl -fsS "$models_url" > "$LOG_DIR/${RUN_NAME}_vllm_models.json" 2>/tmp/cosda_vllm_curl.err; then
      if grep -Fq "$expected_model" "$LOG_DIR/${RUN_NAME}_vllm_models.json"; then
        echo "vLLM ready: $expected_model"
        return 0
      fi
      echo "vLLM endpoint is up but does not list expected model yet: $expected_model"
    fi
    if ! tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
      echo "vLLM session exited while waiting for $expected_model" >&2
      tail -n 160 "$LOG_DIR/${RUN_NAME}_vllm.log" >&2 || true
      return 1
    fi
    sleep 10
  done
  echo "Timed out waiting for vLLM model: $expected_model" >&2
  tail -n 160 "$LOG_DIR/${RUN_NAME}_vllm.log" >&2 || true
  return 1
}

start_vllm() {
  local model="$1"
  local tag="$2"
  if tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
    if [ "$ALLOW_REPLACE_VLLM_SESSION" = "1" ]; then
      tmux kill-session -t "$VLLM_SESSION"
      sleep 3
    else
      echo "Refusing to replace existing tmux session: $VLLM_SESSION" >&2
      echo "Use a unique VLLM_SESSION or set ALLOW_REPLACE_VLLM_SESSION=1 if it is definitely a stale CoSDA session." >&2
      exit 1
    fi
  fi
  local logfile="$LOG_DIR/${RUN_NAME}_vllm_${tag}.log"
  ln -sf "$logfile" "$LOG_DIR/${RUN_NAME}_vllm.log"
  tmux new-session -d -s "$VLLM_SESSION" \
    "cd '$PWD'; source scratch.env 2>/dev/null || true; export PATH='$PATH'; export CUDA_VISIBLE_DEVICES='$VLLM_GPU_IDS' HF_HOME='${HF_HOME:-/mnt/cosda/hf}' TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE:-/mnt/cosda/hf}' VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS='$OMP_NUM_THREADS' MKL_NUM_THREADS='$MKL_NUM_THREADS' TOKENIZERS_PARALLELISM='$TOKENIZERS_PARALLELISM'; vllm serve '$model' --host '$VLLM_HOST' --port '$VLLM_PORT' --dtype '$VLLM_DTYPE' --max-model-len '$VLLM_MAX_MODEL_LEN' --gpu-memory-utilization '$VLLM_GPU_MEMORY_UTILIZATION' --enable-prefix-caching $VLLM_EXTRA_ARGS > '$logfile' 2>&1"
  wait_for_vllm "$model"
}

stop_vllm() {
  if tmux has-session -t "$VLLM_SESSION" 2>/dev/null; then
    tmux kill-session -t "$VLLM_SESSION"
    sleep 5
  fi
}

echo "== CoSDA vLLM phased pipeline =="
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "generation_model=$GEN_MODEL"
echo "judge_model=$JUDGE_MODEL"
echo "resource_profile=$COSDA_RESOURCE_PROFILE"
echo "scenario_jobs=$SCENARIO_JOBS judge_jobs=$JUDGE_JOBS train_jobs=$TRAIN_JOBS gpu_ids=$GPU_IDS gpu_policy=$GPU_POLICY"
echo "gpu_free_thresholds=max_memory_used_mib:$MAX_GPU_MEMORY_USED_MIB max_utilization:$MAX_GPU_UTILIZATION poll_seconds:$GPU_POLL_SECONDS"
echo "vllm_session=$VLLM_SESSION vllm_gpu_ids=$VLLM_GPU_IDS allow_replace_vllm_session=$ALLOW_REPLACE_VLLM_SESSION"
echo "client_concurrency_generation≈$((SCENARIO_JOBS * $(config_value generation.max_workers)))"
echo "client_concurrency_judge≈$((JUDGE_JOBS * $(config_value judge.max_workers)))"
echo "runtime_rps_assumptions=generation:$GENERATION_RPS counterfactual:$COUNTERFACTUAL_RPS judge:$JUDGE_RPS"
echo "omp_threads=$OMP_NUM_THREADS mkl_threads=$MKL_NUM_THREADS tokenizers_parallelism=$TOKENIZERS_PARALLELISM"

python scripts/collect_environment.py | tee "$LOG_DIR/${RUN_NAME}_environment.json"
python scripts/integrity_check.py
"$COSDA" plan-summary --config "$CONFIG" | tee "$LOG_DIR/${RUN_NAME}_plan_summary.json"
python scripts/estimate_runtime.py --config "$CONFIG" --profile g6e_1gpu --gpu-jobs "$TRAIN_JOBS" \
  --generation-rps "$GENERATION_RPS" \
  --counterfactual-rps "$COUNTERFACTUAL_RPS" \
  --judge-rps "$JUDGE_RPS" \
  | tee "$LOG_DIR/${RUN_NAME}_runtime_estimate.json"

"$COSDA" plan-scenario --config "$CONFIG" --run-name "$RUN_NAME" > "$LOG_DIR/${RUN_NAME}_scenario_full.sh"
python scripts/filter_scenario_phase.py "$LOG_DIR/${RUN_NAME}_scenario_full.sh" --phase generation > "$LOG_DIR/${RUN_NAME}_scenario_generation.sh"
python scripts/filter_scenario_phase.py "$LOG_DIR/${RUN_NAME}_scenario_full.sh" --phase judge > "$LOG_DIR/${RUN_NAME}_scenario_judge.sh"

start_vllm "$GEN_MODEL" "generation"
python scripts/preflight_aws.py --config "$CONFIG" --require-gpu --require-imports --check-endpoints --min-free-gib 15 \
  | tee "$LOG_DIR/${RUN_NAME}_preflight_generation.json"
python scripts/run_cell_queue.py "$LOG_DIR/${RUN_NAME}_scenario_generation.sh" \
  --jobs "$SCENARIO_JOBS" \
  --stop-on-failure
stop_vllm

start_vllm "$JUDGE_MODEL" "judge"
python scripts/preflight_aws.py --config "$CONFIG" --require-gpu --require-imports --check-endpoints --min-free-gib 15 \
  | tee "$LOG_DIR/${RUN_NAME}_preflight_judge.json"
python scripts/run_cell_queue.py "$LOG_DIR/${RUN_NAME}_scenario_judge.sh" \
  --jobs "$JUDGE_JOBS" \
  --stop-on-failure
stop_vllm

nvidia-smi || true
"$COSDA" plan-training --config "$CONFIG" --run-name "$RUN_NAME" > "$LOG_DIR/${RUN_NAME}_training.sh"
python scripts/run_command_queue.py "$LOG_DIR/${RUN_NAME}_training.sh" \
  --gpus "$GPU_IDS" \
  --jobs "$TRAIN_JOBS" \
  --gpu-policy "$GPU_POLICY" \
  --max-gpu-memory-used-mib "$MAX_GPU_MEMORY_USED_MIB" \
  --max-gpu-utilization "$MAX_GPU_UTILIZATION" \
  --gpu-poll-seconds "$GPU_POLL_SECONDS" \
  --stop-on-failure

python scripts/export_artifacts.py \
  --runs-dir "$RUNS_DIR" \
  --run-name "$RUN_NAME" \
  --config "$CONFIG" \
  --out-dir "$ARTIFACT_DIR"

echo "Done. Copy artifacts from: $ARTIFACT_DIR"
