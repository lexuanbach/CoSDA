#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scenarios/aws_g6e4xlarge_50gb.yaml}"
RUN_NAME="${RUN_NAME:-classification_pilot_50gb}"
SCENARIO_JOBS="${SCENARIO_JOBS:-1}"
TRAIN_JOBS="${TRAIN_JOBS:-1}"
GPU_IDS="${GPU_IDS:-auto}"
GPU_POLICY="${GPU_POLICY:-wait-free}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
STOP_VLLM_BEFORE_TRAINING="${STOP_VLLM_BEFORE_TRAINING:-0}"
VLLM_TMUX_SESSION="${VLLM_TMUX_SESSION:-cosda_vllm}"

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
  echo "cosda command not found; activate .venv or set COSDA=/path/to/cosda" >&2
  exit 1
fi

echo "== CoSDA g6e.4xlarge 50GB pilot =="
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "runs_dir=$RUNS_DIR"
echo "scenario_jobs=$SCENARIO_JOBS train_jobs=$TRAIN_JOBS gpu_ids=$GPU_IDS"
echo "gpu_policy=$GPU_POLICY max_gpu_memory_used_mib=$MAX_GPU_MEMORY_USED_MIB max_gpu_utilization=$MAX_GPU_UTILIZATION"
echo "stop_vllm_before_training=$STOP_VLLM_BEFORE_TRAINING vllm_tmux_session=$VLLM_TMUX_SESSION"

python scripts/collect_environment.py | tee "$LOG_DIR/${RUN_NAME}_environment.json"
python scripts/preflight_aws.py --config "$CONFIG" --require-gpu --require-imports --check-endpoints --min-free-gib 15 \
  | tee "$LOG_DIR/${RUN_NAME}_preflight.json"
python scripts/integrity_check.py

"$COSDA" plan-summary --config "$CONFIG" | tee "$LOG_DIR/${RUN_NAME}_plan_summary.json"
python scripts/estimate_runtime.py --config "$CONFIG" --profile g6e_1gpu | tee "$LOG_DIR/${RUN_NAME}_runtime_estimate.json"

"$COSDA" plan-scenario --config "$CONFIG" --run-name "$RUN_NAME" > "$LOG_DIR/${RUN_NAME}_scenario.sh"
python scripts/run_cell_queue.py "$LOG_DIR/${RUN_NAME}_scenario.sh" \
  --jobs "$SCENARIO_JOBS" \
  --stop-on-failure

if [ "$STOP_VLLM_BEFORE_TRAINING" = "1" ]; then
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$VLLM_TMUX_SESSION" 2>/dev/null; then
    echo "Stopping vLLM tmux session before training: $VLLM_TMUX_SESSION"
    tmux kill-session -t "$VLLM_TMUX_SESSION"
    sleep 5
  else
    echo "No vLLM tmux session to stop: $VLLM_TMUX_SESSION"
  fi
  nvidia-smi || true
fi

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
