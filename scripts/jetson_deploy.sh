#!/usr/bin/env bash
# Deploy moe-lab to the Jetson and print the exact command to launch a run.
# Usage: JETSON_HOST=user@host scripts/jetson_deploy.sh <run-name> [config-path]
set -euo pipefail

RUN_NAME="${1:?usage: JETSON_HOST=user@host jetson_deploy.sh <run-name> [config.yaml]}"
CONFIG="${2:-configs/base.yaml}"
REMOTE="${JETSON_HOST:?set JETSON_HOST=user@host for your Jetson}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete \
  --exclude '.git/' \
  --exclude 'runs/' \
  --exclude 'data_cache/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "${REPO_ROOT}/" "${REMOTE}:~/moe-lab/"

cat <<EOF
Deployed to ${REMOTE}:~/moe-lab

Start run '${RUN_NAME}' on the Jetson with:

  ssh ${REMOTE} "cd ~/moe-lab && mkdir -p runs/${RUN_NAME} && \
    nohup .venv/bin/python train.py --config ${CONFIG} --name ${RUN_NAME} \
    >> runs/${RUN_NAME}/nohup.log 2>&1 &"

Then follow it with:

  ssh ${REMOTE} "tail -f ~/moe-lab/runs/${RUN_NAME}/nohup.log ~/moe-lab/runs/${RUN_NAME}/telemetry.jsonl"
EOF
