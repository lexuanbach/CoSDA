#!/usr/bin/env bash
set -euo pipefail

SCRATCH="${COSDA_SCRATCH:-/mnt/cosda}"
FORMAT_INSTANCE_STORE=0

for arg in "$@"; do
  case "$arg" in
    --format-instance-store)
      FORMAT_INSTANCE_STORE=1
      ;;
    --scratch=*)
      SCRATCH="${arg#--scratch=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [ "$FORMAT_INSTANCE_STORE" -eq 1 ]; then
  DEVICE="$(
    for dev in /dev/nvme*n1 /dev/xvd? /dev/sd?; do
      [ -b "$dev" ] || continue
      if lsblk -nr -o MOUNTPOINT "$dev" | awk 'NF {found=1} END {exit found ? 0 : 1}'; then
        continue
      fi
      size="$(lsblk -dn -b -o SIZE "$dev" 2>/dev/null | awk 'NR == 1 {print $1}')"
      [ -n "$size" ] || continue
      echo "$size $dev"
    done | sort -nr | head -1 | awk '{print $2}'
  )"
  if [ -z "${DEVICE:-}" ]; then
    echo "No unmounted instance-store disk found; using $SCRATCH on the current filesystem." >&2
    sudo mkdir -p "$SCRATCH"
    sudo chown "$USER":"$USER" "$SCRATCH"
  else
    echo "Formatting and mounting ephemeral instance-store disk $DEVICE at $SCRATCH"
    sudo mkdir -p "$SCRATCH"
    sudo mkfs.ext4 -F "$DEVICE"
    sudo mount "$DEVICE" "$SCRATCH"
    sudo chown "$USER":"$USER" "$SCRATCH"
  fi
else
  sudo mkdir -p "$SCRATCH"
  sudo chown "$USER":"$USER" "$SCRATCH"
fi

mkdir -p \
  "$SCRATCH"/{runs,tmp,hf,torch,pip-cache,cache,vllm,logs,artifacts} \
  "$SCRATCH/hf"/{datasets,hub,transformers}

cat > scratch.env <<EOF
export COSDA_SCRATCH=$SCRATCH
export COSDA_RUNS=$SCRATCH/runs
export COSDA_ARTIFACTS=$SCRATCH/artifacts
export COSDA_LOGS=$SCRATCH/logs
export HF_HOME=$SCRATCH/hf
export HF_HUB_CACHE=$SCRATCH/hf/hub
export HF_DATASETS_CACHE=$SCRATCH/hf/datasets
export TRANSFORMERS_CACHE=$SCRATCH/hf/transformers
export TORCH_HOME=$SCRATCH/torch
export XDG_CACHE_HOME=$SCRATCH/cache
export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export VLLM_CACHE_ROOT=$SCRATCH/vllm
EOF

if [ ! -e runs ]; then
  ln -s "$SCRATCH/runs" runs
fi

echo "Scratch ready at $SCRATCH"
echo "Load it in every shell with: source scratch.env"
