#!/bin/bash
#SBATCH --job-name=grape_detectron2
#SBATCH --output=logs/detectron2/grape_detectron_%j.out
#SBATCH --error=logs/detectron2/grape_detectron_%j.err
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Move to directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Load modules or activate conda environment
module load cuda/12.6
source /home/aja294/conda/etc/profile.d/conda.sh
conda activate detectron2

# Print resource info
echo "Job started on $(hostname) at $(date)"
echo "Using GPU:"
nvidia-smi

# Run training
python scripts/detectron2/train_val/train_detectron.py \
  --config configs/detectron2/detectron2_grape.yaml \
  --output_dir checkpoints/detectron2/5-2 \
  --resume

echo "Job completed at $(date)"