#!/bin/bash
# Submit / preview the clustering+invariance sweep using NERSC module python (conda + pytorch).
set -euo pipefail
module load conda pytorch/2.6.0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: interpreter not found: $PYTHON (load conda first?)" >&2
  exit 1
fi
export NETBURST_PYTHON="$(command -v "$PYTHON")"
exec "$PYTHON" "${SCRIPT_DIR}/submit_clustering_then_invariance_sweep_from_json.py" "$@"
