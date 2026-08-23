#!/usr/bin/env bash
set -euo pipefail

# Opportunistic shared-server launcher. It never steals a busy GPU: training
# starts only after one GPU is below the memory/utilization thresholds for
# several consecutive polls. Run this script with nice/ionice when sharing CPU.

export WAIT_FOR_OTHER_PROJECTS="${WAIT_FOR_OTHER_PROJECTS:-0}"
export SELECTION_JOBS="${SELECTION_JOBS:-1}"
export TRAIN_JOBS="${TRAIN_JOBS:-1}"
export GPU_IDS="${GPU_IDS:-auto}"
export GPU_POLICY="${GPU_POLICY:-wait-free}"
export MAX_GPU_MEMORY_USED_MIB="${MAX_GPU_MEMORY_USED_MIB:-1024}"
export MAX_GPU_UTILIZATION="${MAX_GPU_UTILIZATION:-10}"
export GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
export GPU_IDLE_CONFIRMATIONS="${GPU_IDLE_CONFIRMATIONS:-3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export COSDA_HF_DETERMINISTIC="${COSDA_HF_DETERMINISTIC:-1}"

echo "== CoSDA v2 one-GPU safe launcher =="
echo "TRAIN_JOBS=$TRAIN_JOBS GPU_IDS=$GPU_IDS GPU_POLICY=$GPU_POLICY"
echo "GPU thresholds: mem<=${MAX_GPU_MEMORY_USED_MIB}MiB util<=${MAX_GPU_UTILIZATION}% confirmations=${GPU_IDLE_CONFIRMATIONS} poll=${GPU_POLL_SECONDS}s"
echo "WAIT_FOR_OTHER_PROJECTS=$WAIT_FOR_OTHER_PROJECTS"
echo "CPU threads: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "This launcher does not stop, kill, reboot, or reserve a GPU that is already in use."

bash scripts/run_v2_hf_pilot.sh
