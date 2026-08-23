#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y git git-lfs python3-venv build-essential tmux htop jq
git lfs install

scripts/setup_aws_scratch.sh "$@"
source scratch.env

python3 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements-aws.txt

python scripts/prepare_raw_data.py
python scripts/build_manifest.py
python scripts/collect_environment.py

echo "Bootstrap complete. Activate with: . .venv/bin/activate"
