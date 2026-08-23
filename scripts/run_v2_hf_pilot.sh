#!/usr/bin/env bash
set -euo pipefail

CONFIG_NEWS="${CONFIG_NEWS:-configs/scenarios/aws_g6e4xlarge_vllm_revised_select1_v2_news_swa.yaml}"
CONFIG_SENTIMENT="${CONFIG_SENTIMENT:-configs/scenarios/aws_g6e4xlarge_vllm_revised_select1_v2_sentiment_amh_hau_yor.yaml}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-aws_vllm_revised_select1_20260520}"
RUN_NAME="${RUN_NAME:-aws_vllm_revised_select1_v2_pilot_20260521}"
SELECTION_JOBS="${SELECTION_JOBS:-2}"
TRAIN_JOBS="${TRAIN_JOBS:-4}"
GPU_IDS="${GPU_IDS:-auto}"
GPU_POLICY="${GPU_POLICY:-wait-free}"
MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-1}"
WAIT_FOR_OTHER_PROJECTS="${WAIT_FOR_OTHER_PROJECTS:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-120}"
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

wait_for_other_projects() {
  if [ "$WAIT_FOR_OTHER_PROJECTS" != "1" ] && [ "$WAIT_FOR_OTHER_PROJECTS" != "true" ]; then
    return
  fi
  echo "Waiting for non-CoSDA GPU/CPU jobs to finish before HF pilot."
  while true; do
    active=""
    if command -v tmux >/dev/null 2>&1; then
      active="$active"$'\n'"$(tmux ls 2>/dev/null | grep -E 'dm_seq2seq_4gpu|dm_cpu_bootstrap50k' || true)"
    fi
    active="$active"$'\n'"$(pgrep -af 'DeltaMerge-LowRes/.+train_task_delta|scripts/paired_bootstrap.py' || true)"
    if [ -z "$(printf '%s' "$active" | sed '/^[[:space:]]*$/d')" ]; then
      echo "No blocking non-CoSDA jobs detected."
      return
    fi
    echo "Still waiting at $(date -u +%Y-%m-%dT%H:%M:%SZ):"
    printf '%s\n' "$active" | sed '/^[[:space:]]*$/d' | head -20
    sleep "$WAIT_POLL_SECONDS"
  done
}

echo "== CoSDA rerank-v2 HF pilot =="
echo "run_name=$RUN_NAME"
echo "source_run_name=$SOURCE_RUN_NAME"
echo "configs=$CONFIG_NEWS $CONFIG_SENTIMENT"
echo "pilot_cells=news_topic_classification/swa sentiment_classification/{amh,hau,yor}"
echo "baselines=naive alpagasus_style deita_style cosda_equal_budget cosda_rerank_v2"
echo "train_jobs=$TRAIN_JOBS gpu_ids=$GPU_IDS gpu_policy=$GPU_POLICY"
echo "gpu_free_thresholds=max_memory_used_mib:$MAX_GPU_MEMORY_USED_MIB max_utilization:$MAX_GPU_UTILIZATION poll_seconds:$GPU_POLL_SECONDS idle_confirmations:$GPU_IDLE_CONFIRMATIONS"
echo "hf_deterministic=$HF_DETERMINISTIC"
echo "No generation, counterfactual, judge, stop, reboot, or process-kill actions will be run."

python scripts/collect_environment.py | tee "$LOG_DIR/${RUN_NAME}_environment.json"
python scripts/integrity_check.py

wait_for_other_projects

python scripts/clone_run_inputs.py \
  --runs-dir "$RUNS_DIR" \
  --source-run "$SOURCE_RUN_NAME" \
  --target-run "$RUN_NAME"

SELECTION_ALL="$LOG_DIR/${RUN_NAME}_selection_all.sh"
TRAINING_ALL="$LOG_DIR/${RUN_NAME}_training_all.sh"
: > "$SELECTION_ALL"
: > "$TRAINING_ALL"

for CONFIG in "$CONFIG_NEWS" "$CONFIG_SENTIMENT"; do
  config_tag="$(basename "$CONFIG" .yaml)"
  "$COSDA" plan-summary --config "$CONFIG" | tee "$LOG_DIR/${RUN_NAME}_${config_tag}_plan_summary.json"
  "$COSDA" plan-scenario --config "$CONFIG" --run-name "$RUN_NAME" > "$LOG_DIR/${RUN_NAME}_${config_tag}_scenario_full.sh"
  python scripts/filter_scenario_phase.py "$LOG_DIR/${RUN_NAME}_${config_tag}_scenario_full.sh" --phase selection > "$LOG_DIR/${RUN_NAME}_${config_tag}_selection.sh"
  cat "$LOG_DIR/${RUN_NAME}_${config_tag}_selection.sh" >> "$SELECTION_ALL"

  python scripts/validate_replay_inputs.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --mode replay-inputs \
    | tee "$LOG_DIR/${RUN_NAME}_${config_tag}_preselection_input_validation.json" || true

  PLAN_TRAINING_ARGS=(--config "$CONFIG" --run-name "$RUN_NAME")
  if [ "$HF_DETERMINISTIC" = "1" ] || [ "$HF_DETERMINISTIC" = "true" ]; then
    PLAN_TRAINING_ARGS+=(--deterministic)
  fi
  "$COSDA" plan-training "${PLAN_TRAINING_ARGS[@]}" > "$LOG_DIR/${RUN_NAME}_${config_tag}_training.sh"
  cat "$LOG_DIR/${RUN_NAME}_${config_tag}_training.sh" >> "$TRAINING_ALL"
done

python scripts/run_cell_queue.py "$SELECTION_ALL" \
  --jobs "$SELECTION_JOBS" \
  --gpus "" \
  --log-dir "$LOG_DIR/${RUN_NAME}_selection_queue_logs" \
  --stop-on-failure

for CONFIG in "$CONFIG_NEWS" "$CONFIG_SENTIMENT"; do
  config_tag="$(basename "$CONFIG" .yaml)"
  python scripts/validate_replay_inputs.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --mode replay-inputs \
    | tee "$LOG_DIR/${RUN_NAME}_${config_tag}_hf_input_validation.json"
done

python scripts/run_command_queue.py "$TRAINING_ALL" \
  --gpus "$GPU_IDS" \
  --jobs "$TRAIN_JOBS" \
  --gpu-policy "$GPU_POLICY" \
  --max-gpu-memory-used-mib "$MAX_GPU_MEMORY_USED_MIB" \
  --max-gpu-utilization "$MAX_GPU_UTILIZATION" \
  --gpu-poll-seconds "$GPU_POLL_SECONDS" \
  --idle-confirmations "$GPU_IDLE_CONFIRMATIONS" \
  --log-dir "$LOG_DIR/${RUN_NAME}_hf_queue_logs" \
  --stop-on-failure

for CONFIG in "$CONFIG_NEWS" "$CONFIG_SENTIMENT"; do
  config_tag="$(basename "$CONFIG" .yaml)"
  python scripts/validate_replay_inputs.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --mode hf-results \
    | tee "$LOG_DIR/${RUN_NAME}_${config_tag}_hf_result_validation.json"
done

python scripts/export_artifacts.py \
  --runs-dir "$RUNS_DIR" \
  --run-name "$RUN_NAME" \
  --config "$CONFIG_NEWS" \
  --out-dir "$ARTIFACT_DIR"

echo "Done. Artifacts are in: $ARTIFACT_DIR"
