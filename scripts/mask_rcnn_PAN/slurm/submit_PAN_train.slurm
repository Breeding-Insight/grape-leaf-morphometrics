#!/bin/bash
#SBATCH --job-name=mask_rcnn_PAN_train
#SBATCH --output=logs/mask_rcnn_PAN/mask_rcnn_PAN_%j.out
#SBATCH --error=logs/mask_rcnn_PAN/mask_rcnn_PAN_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=regular

# Activate conda environment
source $HOME/.bashrc
conda activate pytorch-cuda12.6

# General GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,garbage_collection_threshold:0.8"

# CPU threading optimizations (use only what is allocated)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Network optimizations (disable IB if not needed)
export NCCL_P2P_LEVEL=NVL
export NCCL_SOCKET_IFNAME=^lo,docker
export NCCL_IB_DISABLE=1

# General performance settings
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=0

# Print job information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "GPU information:"
nvidia-smi

# Navigate to project directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Run training script
python /workdir/data/grape/grape_pheno/grape-leaf-morphometrics/scripts/mask_rcnn_PAN/train_val/train_grape_mask_rcnn_PAN.py

echo "Job completed at $(date)" 