#!/bin/bash
# Submit Chronos2 CKA, then clustering+invariance with Slurm afterok dependency (see ../submit_chronos2_cka_then_clustering.py).
set -euo pipefail
module load conda pytorch/2.6.0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# After conda: use `python` from PATH (not the login node's interpreter). Override: PYTHON=/path/to/python
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: interpreter not found: $PYTHON (load conda first?)" >&2
  exit 1
fi
export NETBURST_PYTHON="$(command -v "$PYTHON")"
exec "$PYTHON" "${SCRIPT_DIR}/../submit_chronos2_cka_then_clustering.py" "$@"
