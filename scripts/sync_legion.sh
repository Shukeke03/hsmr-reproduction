#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Only code; excluded directories hold data and experiment results.
rsync -az --exclude='.git' --exclude='__pycache__' --exclude='*.egg-info' \
  --exclude='data_inputs/' --exclude='data_outputs/' --exclude='.local/' \
  ./ legion:/home/shuxing/workspace/projects/hsmr-reproduction/
