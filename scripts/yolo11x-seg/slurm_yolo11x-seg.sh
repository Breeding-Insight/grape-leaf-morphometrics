#!/bin/bash
#SBATCH --job-name=yolo11x_seg_train
#SBATCH --output=logs/yolo11x_seg/yolo11x_seg_%j.out
#SBATCH --error=logs/yolo11x_seg/yolo11x_seg_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --partition=regular

# ============================================================================
# YOLO11X-SEG TRAINING SLURM SCRIPT
# ============================================================================
# This script is optimized for training YOLO11x-seg models for grape leaf segmentation
# ============================================================================

# Activate conda environment
source $HOME/.bashrc
conda activate yolo11

# ============================================================================
# GPU AND MEMORY OPTIMIZATIONS FOR YOLO11X-SEG
# ============================================================================

# General GPU settings
export CUDA_VISIBLE_DEVICES=0
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache for YOLO11x

# Enhanced memory optimizations for YOLO11x-seg
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,expandable_segments:True,garbage_collection_threshold:0.8"

# YOLO-specific optimizations
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
echo "YOLO11X-SEG TRAINING"
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
echo "Ultralytics version: $(python -c 'import ultralytics; print(ultralytics.__version__)')"
echo "============================================================================"

# ============================================================================
# DIRECTORY SETUP AND NAVIGATION
# ============================================================================

# Navigate to project directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Create log directories if they don't exist
mkdir -p logs/yolo11x_seg

echo "Working directory: $(pwd)"
echo "============================================================================"

# ============================================================================
# MODEL ARCHITECTURE INFORMATION
# ============================================================================

echo "Model Architecture: YOLO11x-seg"
echo "Features:"
echo "  - YOLO11x segmentation model"
echo "  - Optimized for grape leaf segmentation"
echo "  - Early stopping with patience=15 epochs"
echo "  - Segmentation-specific loss weights"
echo "  - Retina masks for high-resolution segmentation"
echo "  - Mask overlap support for multi-instance segmentation"
echo "============================================================================"

# ============================================================================
# TRAINING EXECUTION
# ============================================================================

echo "Starting YOLO11x-seg training..."
echo "Training script: scripts/yolo11x-seg/train_val/train_val_yolo11x-seg.py"
echo "Data config: data/annotations/yolo11x-seg/data.yaml"
echo "============================================================================"

# Run the YOLO11x-seg training script
python scripts/yolo11x-seg/train_val/train_val_yolo11x-seg.py \
    --data data/annotations/yolo11x-seg/data.yaml \
    --output-dir checkpoints/train_yolo11x_seg_7-31 \
    --model yolo11x-seg.pt \
    --epochs 100 \
    --batch-size 8 \
    --img-size 640 \
    --workers 8 \
    --device 0 \
    --optimizer AdamW \
    --patience 15 \
    --min-delta 0.001 \
    --amp \
    --save-period 5 \
    --verbose

# Capture the exit code
EXIT_CODE=$?

# ============================================================================
# JOB COMPLETION AND CLEANUP
# ============================================================================

echo "============================================================================"
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: YOLO11x-seg training completed successfully!"
    echo "Check the checkpoints/train_yolo11x_seg_7-31 directory for saved models."
    echo "Best model: checkpoints/train_yolo11x_seg_7-31/weights/yolo11x-seg_best.pt"
    echo "Final model: checkpoints/train_yolo11x_seg_7-31/weights/yolo11x-seg_final.pt"
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
