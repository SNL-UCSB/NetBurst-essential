#!/bin/bash
#SBATCH -q premium
#SBATCH --account=<SLURM_ACCOUNT>
#SBATCH --job-name=netburst_pre_ibgbi_all
# Parent walltime limits the whole job; nested bash *.sl #SBATCH lines do not apply.
#SBATCH --time=2-00:00:00
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
# PARQUET_ROOT in each child script is <split_output_dir>/ibgbi_train (data lives outside this repo).
# SAVE_DIR: NetBurst checkpoints only under this repo:
#   src/train/checkpoints/netburst_ibgbi_manifest/NetBurstIBGBI_*  (see each child).
# Code: torchrun pretrain_twin.py from NetBurst-essential/src/train (see src/train/README.md).

srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_ip_1s_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_subnet_1s_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_service_100ms_main.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_service_100ms_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_ip_1s_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_subnet_1s_mawi.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_rne_netflow_ip.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_rne_netflow_service.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_perfsonar_p99_thresh3.sl &
srun --exclusive -N 1 -n 1 bash netburst_pretrain_ibgbi_manifest_ping_latency_3min_thresh4.sl &

wait
echo "All manifest-based Chronos pretraining jobs completed (10 datasets)."
