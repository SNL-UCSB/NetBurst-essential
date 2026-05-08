#!/bin/bash
# Load NERSC modules, validate generated Slurm (preview), then submit.
# Usage:
#   ./run_invariance_sweep_slurm.sh [JSON_FILE]
# Default JSON: example_cluster_invariance_multi_node_per_run_parallel_k.json (full sweep)
#
# Environment:
#   PYTHON   Override python (default: python3.11)
#   PREVIEW_ONLY=1   Only run preview + checks, do not sbatch

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

JSON="${1:-example_cluster_invariance_multi_node_per_run_parallel_k.json}"
PYTHON="${PYTHON:-python3.11}"
TMP_ROOT="${TMPDIR:-/tmp}/netburst_invariance_$$"
PREV_DIR="${TMP_ROOT}/preview"

if [[ ! -f "$JSON" ]]; then
  echo "Config not found: $JSON" >&2
  exit 1
fi

echo "==> Loading modules (conda, pytorch)..."
module load conda 2>/dev/null || true
module load pytorch/2.6.0 2>/dev/null || {
  echo "WARN: pytorch/2.6.0 not found; trying without explicit pytorch module." >&2
}

echo "==> Python: $($PYTHON --version 2>&1)"
echo "==> Preview: $JSON -> $PREV_DIR"
mkdir -p "$PREV_DIR"
"$PYTHON" submit_cluster_invariance_sweep_from_json.py --json "$JSON" --mode preview --tmp-dir "$PREV_DIR"

SLURM_FILE=$(echo "$PREV_DIR"/*_preview.slurm | head -1)
if [[ ! -f "$SLURM_FILE" ]]; then
  echo "No preview slurm under $PREV_DIR" >&2
  exit 1
fi

echo "==> Checking $SLURM_FILE for staging + NB_* + env python..."
if ! grep -q 'export STAGE_DIR' "$SLURM_FILE"; then
  echo "FAIL: missing export STAGE_DIR" >&2
  exit 1
fi
if ! grep -q 'export NETBURST_STAGE_DIR' "$SLURM_FILE"; then
  echo "FAIL: missing export NETBURST_STAGE_DIR" >&2
  exit 1
fi
# Sequential: NB_CLUSTERING=... ; parallel staging: literal --clustering .../netburst_invariance_${SLURM_JOB_ID}...
if ! grep -qE 'NB_CLUSTERING(_[0-9]+)?=' "$SLURM_FILE"; then
  if ! grep -q 'cluster_analysis.py.*--clustering.*netburst_invariance' "$SLURM_FILE"; then
    echo "FAIL: missing NB_CLUSTERING= or literal --clustering ... netburst_invariance path" >&2
    exit 1
  fi
fi
if ! grep -q 'NB_TSFRESH=' "$SLURM_FILE"; then
  if ! grep -q 'cluster_analysis.py.*--tsfresh.*netburst_invariance' "$SLURM_FILE"; then
    echo "FAIL: missing NB_TSFRESH= or literal --tsfresh ... netburst_invariance path" >&2
    exit 1
  fi
fi
if ! grep -q 'env STAGE_DIR=.*NETBURST_STAGE_DIR=.*python3 cluster_analysis.py' "$SLURM_FILE"; then
  echo "FAIL: missing env STAGE_DIR=... NETBURST_STAGE_DIR=... python3 cluster_analysis.py" >&2
  exit 1
fi
if ! grep -q 'netburst-stage-dir' "$SLURM_FILE"; then
  echo "FAIL: missing --netburst-stage-dir (live STAGE_DIR for this job)" >&2
  exit 1
fi
echo "OK: STAGE_DIR + NETBURST_STAGE_DIR export, staged argv (NB_* or literal netburst paths), env-wrapped python3, and --netburst-stage-dir present."

if [[ "${PREVIEW_ONLY:-0}" == "1" ]]; then
  echo "PREVIEW_ONLY=1 -> skip sbatch"
  echo "Preview file: $SLURM_FILE"
  exit 0
fi

echo "==> Submitting Slurm job..."
# Avoid forwarding a stray STAGE_DIR from this shell into sbatch (Slurm exports submit env by default).
unset STAGE_DIR NETBURST_STAGE_DIR 2>/dev/null || true
OUT="$("$PYTHON" submit_cluster_invariance_sweep_from_json.py --json "$JSON" --mode slurm --tmp-dir "${TMP_ROOT}/submit" 2>&1)" || {
  echo "$OUT" >&2
  exit 1
}
echo "$OUT"
rm -rf "$TMP_ROOT" 2>/dev/null || true
echo "Done."
