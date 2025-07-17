#!/bin/bash
#SBATCH --job-name=hybrid_resnext_panet_train
#SBATCH --output=logs/mask_rcnn_nightly/hybrid_resnext_panet_%j.out
#SBATCH --error=logs/mask_rcnn_nightly/hybrid_resnext_panet_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=regular

# ============================================================================
# HYBRID RESNEXT-101 32x8d PANET MASK R-CNN TRAINING SLURM SCRIPT
# ============================================================================
# This script is optimized for training the hybrid ResNeXt-101 32x8d PANet model
# which combines pretrained ResNeXt-101 32x8d backbone with custom PANet aggregation
# ============================================================================

# Activate conda environment
source $HOME/.bashrc
conda activate pytorch_nightly_cu129

# ============================================================================
# GPU AND MEMORY OPTIMIZATIONS FOR RESNEXT-101 32x8d
# ============================================================================

# General GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=4294967296  # Increased to 4GB for ResNeXt

# Enhanced memory optimizations for ResNeXt-101 32x8d
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8"

# ResNeXt-specific optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_CUDNN_V8_API_DISABLED=0

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

# ============================================================================
# JOB INFORMATION AND MONITORING
# ============================================================================

echo "============================================================================"
echo "HYBRID RESNEXT-101 32x8d PANET MASK R-CNN TRAINING"
echo "============================================================================"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK"
echo "Allocated Memory: $SLURM_MEM_PER_NODE"
echo "Allocated GPU: $CUDA_VISIBLE_DEVICES"
echo "============================================================================"

# Display GPU information
echo "GPU Information:"
nvidia-smi
echo "============================================================================"

# Display system information
echo "System Information:"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "Disk space: $(df -h . | tail -1 | awk '{print $4}') available"
echo "============================================================================"

# ============================================================================
# ENVIRONMENT VERIFICATION
# ============================================================================

echo "Environment Verification:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(f"PyTorch {torch.__version__}")')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "GPU name: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "============================================================================"

# ============================================================================
# DIRECTORY SETUP AND NAVIGATION
# ============================================================================

# Navigate to project directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Create log directories if they don't exist
mkdir -p logs/mask_rcnn_nightly

echo "Working directory: $(pwd)"
echo "============================================================================"

# ============================================================================
# MODEL ARCHITECTURE INFORMATION
# ============================================================================

echo "Model Architecture: Hybrid ResNeXt-101 32x8d PANet Mask R-CNN"
echo "Features:"
echo "  - Pretrained ResNeXt-101 32x8d backbone (torchvision)"
echo "  - Grouped convolutions (groups=32, width_per_group=8)"
echo "  - Custom PANet path aggregation"
echo "  - Enhanced mask predictor (28x28 resolution)"
echo "  - Optimized anchor generation for leaf detection"
echo "  - Superior feature representation with grouped convolutions"
echo "============================================================================"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

echo "Starting Hybrid ResNeXt-101 32x8d PANet training..."
echo "Training script: scripts/mask_rcnn_nightly/train_val/train_grape_mask_rcnn.py"
echo "============================================================================"

# Run the hybrid ResNeXt training script
python -m scripts.mask_rcnn_nightly.train_val.train_grape_mask_rcnn

# Capture the exit code
EXIT_CODE=$?

# ============================================================================
# JOB COMPLETION AND CLEANUP
# ============================================================================

echo "============================================================================"
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Hybrid ResNeXt-101 32x8d PANet training completed successfully!"
    echo "Check the checkpoint directory for saved models."
else
    echo "ERROR: Training failed with exit code $EXIT_CODE"
    echo "Check the error logs for details."
fi

echo "============================================================================"

# Final GPU status
echo "Final GPU status:"
nvidia-smi
echo "============================================================================"

# Exit with the same code as the training script
exit $EXIT_CODE
