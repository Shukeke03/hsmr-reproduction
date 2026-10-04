#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export HSMR_INPUTS_ROOT="${HSMR_INPUTS_ROOT:-$HOME/workspace/datasets/HSMR}"
export HSMR_OUTPUTS_ROOT="${HSMR_OUTPUTS_ROOT:-$HOME/workspace/checkpoints/HSMR}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# WSL GPU tools are not in non-interactive SSH PATH by default.
export PATH="/usr/lib/wsl/lib:$PATH"
python_bin="${MINIFORGE_PREFIX:-$HOME/miniforge3}/envs/hsmr/bin/python"
checkpoint="${HSMR_CHECKPOINT:-$(find "$HSMR_INPUTS_ROOT/released_models" -type f -name hsmr.ckpt -print -quit 2>/dev/null || true)}"
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo 'Missing released hsmr.ckpt. Run scripts/prepare_data.py --assets on legion.' >&2
  exit 1
fi
if [[ ! -f "$HSMR_INPUTS_ROOT/body_models/skel/skel_male.pkl" ]]; then
  echo 'Missing licensed SKEL v1.1.1: install it under body_models/skel.' >&2
  exit 1
fi
exec "$python_bin" exp/run_train.py exp=hsmr/legion_finetune "ckpt_path=$checkpoint" "$@"
