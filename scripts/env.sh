#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export NDBC_ROOT="$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if [ -x "$HOME/miniconda3/envs/EquSQL/bin/python" ]; then
  export PYTHON_BIN="$HOME/miniconda3/envs/EquSQL/bin/python"
elif [ -x "$HOME/anaconda3/envs/EquSQL/bin/python" ]; then
  export PYTHON_BIN="$HOME/anaconda3/envs/EquSQL/bin/python"
else
  export PYTHON_BIN="${PYTHON_BIN:-python}"
fi
