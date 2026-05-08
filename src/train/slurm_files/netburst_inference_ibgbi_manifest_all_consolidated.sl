#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_inf_ibgbi_all
#SBATCH --time=10:00:00
#SBATCH --licenses=scratch
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --qos=premium

module load conda
module load pytorch/2.6.0

cd <repo-root>/src/train/slurm_files
# PARQUET_ROOT in each child matches the manifest ibgbi_test splits (same paths as NetBurst child scripts).
# MODEL_PATH in each child: NetBurst checkpoints under NetBurst src/train (NetBurstIBGBI_*), matching
# corresponding pretrain SAVE_DIR values from this repo's train Slurm templates.
# Inference code: torchrun ar_predict.py from NetBurst-essential/src/train (see src/train/README.md).

srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_ip_1s_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_subnet_1s_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_service_100ms_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_service_100ms_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_ip_1s_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_subnet_1s_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_rne_netflow_ip.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_rne_netflow_service.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_perfsonar_p99_thresh3.sl &
srun --exclusive -N 1 -n 1 bash netburst_inference_ibgbi_manifest_ping_latency_3min_thresh4.sl &

wait
echo "All manifest-based Chronos inference jobs completed (10 datasets)."
