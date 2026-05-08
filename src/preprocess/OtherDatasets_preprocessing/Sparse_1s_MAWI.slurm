#!/bin/bash
#SBATCH -q premium
#SBATCH -C cpu
#SBATCH --nodes=16
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=pinot_sparse_nograph
#SBATCH --ntasks=16
#SBATCH --time=2:00:00

module load conda
module load pytorch/2.6.0
PINOT_SCRIPT="${PINOT_SCRIPT:-<repo-root>/src/preprocess/OtherDatasets_preprocessing/NonGraphSparseTimeseries.py}"
EXTRA=()
if [[ -n "${PINOT_INPUT_PARQUET_MAWI:-}" ]]; then
  EXTRA+=(--input_parquet "${PINOT_INPUT_PARQUET_MAWI}")
elif [[ -n "${PINOT_INPUT_PARQUET:-}" ]]; then
  EXTRA+=(--input_parquet "${PINOT_INPUT_PARQUET}")
fi
# sbatch --export=ALL from upstream can inherit SLURM_CPUS_PER_TASK; drop so srun agrees with this job.
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK 2>/dev/null || true
srun python3 "${PINOT_SCRIPT}" "${EXTRA[@]}" --bin_ms 1000 --output_dir <data-root>/sparse_timeseries_1s_MAWI --finetuning --min_seq_len 3 --min_bursts 3
