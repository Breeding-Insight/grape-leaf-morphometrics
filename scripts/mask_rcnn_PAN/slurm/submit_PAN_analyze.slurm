#!/bin/bash
#SBATCH --job-name=mask_rcnn_analyze
#SBATCH --output=logs/mask_rcnn_PAN/mask_rcnn_analyze_%j.out
#SBATCH --error=logs/mask_rcnn_PAN/mask_rcnn_analyze_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=regular

# Activate conda environment
source $HOME/.bashrc
conda activate pytorch-cuda12.6

echo "Conda environment activated, CUDA = 12.6"

# General performance settings
export PYTHONUNBUFFERED=1
export MALLOC_TRIM_THRESHOLD_=0

# Print job information
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Current working directory: $(pwd)"

which python
python -c "import sys; print(sys.executable)"

# Navigate to project directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Add project root to PYTHONPATH
export PYTHONPATH=/workdir/data/grape/grape_pheno/grape-leaf-morphometrics:$PYTHONPATH

# Run analysis script
python /workdir/data/grape/grape_pheno/grape-leaf-morphometrics/scripts/mask_rcnn_PAN/test/analyze_PAN_maskrcnn.py

echo "Job completed at $(date)" 