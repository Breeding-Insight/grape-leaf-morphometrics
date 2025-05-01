#!/bin/bash
#SBATCH --job-name=mask_rcnn_single
#SBATCH --output=mask_rcnn_%j.out
#SBATCH --error=mask_rcnn_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14          # Since we use 14 workers total
#SBATCH --gres=gpu:RTX6000:1        # Single GPU
#SBATCH --mem=128G                  # Adjusted for single GPU
#SBATCH --partition=regular

# Email notifications
#SBATCH --mail-user=aja294@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Activate conda environment
source $HOME/.bashrc
conda activate pytorch_nightly_cuda12.6-env 

# Single GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_PATH=/scratch/$USER/.cuda_cache
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:4096,garbage_collection_threshold:0.9"

# AMD CPU optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export GOMP_CPU_AFFINITY="0-15"

# Performance tuning
export MALLOC_TRIM_THRESHOLD_=0
export PYTHONUNBUFFERED=1

# Job info
echo "=== Job Information ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi -L
echo "======================="

# Training launch
cd /workdir/data/grape/grape_pheno/grape_peduncle

# Activate conda env
conda activate pytorch_nightly_cuda12.6-env

# Training command
python scripts/mask-rcnn/train_val/train_grape_mask-rcnn.py

echo "=== Training Completed ==="
echo "End: $(date)"
echo "========================="
