#!/bin/bash
#SBATCH --job-name=resnext101_detectron2
#SBATCH --output=logs/detectron2_v2/X101-FPN_%j.out
#SBATCH --error=logs/detectron2_v2/X101-FPN_%j.err
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "Job started on $(hostname) at $(date)"

# Move to directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Load modules or activate conda environment
source /home/aja294/conda/etc/profile.d/conda.sh
conda activate detectron2

# Print resource info
echo "Conda environment activated as of $(date)"
echo "Using GPU:"
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs/detectron2_v2

# Run cross-validation training
echo "Starting 5-fold cross-validation for leaf segmentation..."
python scripts/detectron2_v2/train_val/cross_validation_wrapper.py \
  --config scripts/detectron2_v2/train_val/X101-FPN/config_detectron2.yaml \
  --full_dataset "data/annotations/coco100/full_dataset.json" \
  --k_folds 5 \
  --output_dir "checkpoints/detectron2_v2/resnext101_fpn_$(date +%Y%m%d_%H%M%S)" \
  --random_state 42

echo "Cross-validation job completed at $(date)"
echo "Results saved in cv_results directory" 
