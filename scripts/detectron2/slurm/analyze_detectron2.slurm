#!/bin/bash
#SBATCH --job-name=detectron2_analysis
#SBATCH --output=logs/detectron2/detectron2_analysis_%j.out
#SBATCH --error=logs/detectron2/detectron2_analysis_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1

cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

module load cuda/12.6
source /home/aja294/conda/etc/profile.d/conda.sh
conda activate detectron2

python scripts/detectron2/test/analyze_detectron2.py \
    --results_dir data/processed/mask-rcnn_test

echo "Detectron2 analysis complete!" 