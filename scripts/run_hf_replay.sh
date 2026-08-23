#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/scenarios/aws_g6e4xlarge_vllm_revised.yaml}"
RUN_NAME="${RUN_NAME:-aws_vllm_revised_full_20260520}"
TRAIN_JOBS="${TRAIN_JOBS:-0}"
GPU_IDS="${GPU_IDS:-auto}"
GPU_POLICY="${GPU_POLICY:-wait-free}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-20}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
COSDA_HF_DETERMINISTIC="${COSDA_HF_DETERMINISTIC:-1}"
HF_DETERMINISTIC="${HF_DETERMINISTIC:-$COSDA_HF_DETERMINISTIC}"

export OMP_NUM_THREADS MKL_NUM_THREADS TOKENIZERS_PARALLELISM COSDA_HF_DETERMINISTIC

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

echo "== CoSDA deterministic HF replay =="
echo "config=$CONFIG"
echo "run_name=$RUN_NAME"
echo "train_jobs=$TRAIN_JOBS gpu_ids=$GPU_IDS gpu_policy=$GPU_POLICY"
echo "gpu_free_thresholds=max_memory_used_mib:$MAX_GPU_MEMORY_USED_MIB max_utilization:$MAX_GPU_UTILIZATION poll_seconds:$GPU_POLL_SECONDS"
echo "hf_deterministic=$HF_DETERMINISTIC"
echo "No generation, counterfactual, or judge calls will be run."

python scripts/collect_environment.py | tee "$LOG_DIR/${RUN_NAME}_hf_replay_environment.json"
python scripts/integrity_check.py
nvidia-smi || true
python scripts/validate_replay_inputs.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --mode replay-inputs \
  | tee "$LOG_DIR/${RUN_NAME}_hf_replay_input_validation.json"

PLAN_TRAINING_ARGS=(--config "$CONFIG" --run-name "$RUN_NAME")
if [ "$HF_DETERMINISTIC" = "1" ] || [ "$HF_DETERMINISTIC" = "true" ]; then
  PLAN_TRAINING_ARGS+=(--deterministic)
fi
"$COSDA" plan-training "${PLAN_TRAINING_ARGS[@]}" > "$LOG_DIR/${RUN_NAME}_hf_replay_training.sh"
python scripts/run_command_queue.py "$LOG_DIR/${RUN_NAME}_hf_replay_training.sh" \
  --gpus "$GPU_IDS" \
  --jobs "$TRAIN_JOBS" \
  --gpu-policy "$GPU_POLICY" \
  --max-gpu-memory-used-mib "$MAX_GPU_MEMORY_USED_MIB" \
  --max-gpu-utilization "$MAX_GPU_UTILIZATION" \
  --gpu-poll-seconds "$GPU_POLL_SECONDS" \
  --stop-on-failure
python scripts/validate_replay_inputs.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME" \
  --mode hf-results \
  | tee "$LOG_DIR/${RUN_NAME}_hf_replay_result_validation.json"

python scripts/export_artifacts.py \
  --runs-dir "$RUNS_DIR" \
  --run-name "$RUN_NAME" \
  --config "$CONFIG" \
  --out-dir "$ARTIFACT_DIR"

echo "Done. Copy replay artifacts from: $ARTIFACT_DIR"
