#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
miniforge_prefix="${MINIFORGE_PREFIX:-$HOME/miniforge3}"
conda_bin="$miniforge_prefix/bin/conda"
if [[ ! -x "$conda_bin" ]]; then
  echo "Miniforge not found at $miniforge_prefix. Set MINIFORGE_PREFIX." >&2
  exit 1
fi
if [[ ! -x "$miniforge_prefix/envs/hsmr/bin/python" ]]; then
  "$conda_bin" env create -f environment.yml
fi
python_bin="$miniforge_prefix/envs/hsmr/bin/python"
if [[ "$(uname -s)" == Linux ]]; then
  "$python_bin" -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
else
  "$python_bin" -m pip install torch==2.3.1 torchvision==0.18.1
fi
"$python_bin" -m pip install -r environments/requirements-common.txt
"$python_bin" -m pip install -e .
"$python_bin" -m pip install -e thirdparty/SKEL
"$python_bin" -m pip check
"$python_bin" scripts/check_environment.py
