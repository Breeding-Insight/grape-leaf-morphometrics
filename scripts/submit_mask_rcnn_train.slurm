#!/bin/bash
#SBATCH --job-name=mask_rcnn_train
#SBATCH --output=mask_rcnn_%j.out
#SBATCH --error=mask_rcnn_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX6000:1      # Updated to match your cluster's GPU specification
#SBATCH --mem=128G
#SBATCH --partition=regular

# Email notifications
#SBATCH --mail-user=aja294@cornell.edu
#SBATCH --mail-user=maw396@cornell.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_90

# Activate conda environment
source $HOME/.bashrc
conda activate torch_nightly_cuda12.6-env  # Replace with your environment name

# RTX 6000 Ada-specific optimizations
export CUDA_VISIBLE_DEVICES=0
export CUDA_AUTO_BOOST=1
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache (increased for 49GB VRAM)
export CUDA_DEVICE_MAX_CONNECTIONS=32  # Increased for Ada architecture

# Memory optimizations for 49GB VRAM
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,garbage_collection_threshold:0.8"

# Clock optimizations based on detected max clocks (3105 MHz)
# Your GPU is already in P0 state with high clocks (2505 MHz)
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# CPU threading optimizations
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

# PCIe 4.0 x16 optimizations (detected from specs)
export NCCL_P2P_LEVEL=NVL  # Use NVLink protocol for P2P
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

# Navigate to your project directory
cd /workdir/data/grape/grape_pheno/grape_leaf_metrics

# Run your training script
python /workdir/data/grape/grape_pheno/grape_leaf_metrics/scripts/train_grape_mask-rcnn_cuda.py

echo "Job completed at $(date)"
