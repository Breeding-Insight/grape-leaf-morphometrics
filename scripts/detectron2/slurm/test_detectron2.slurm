#!/bin/bash
#SBATCH --job-name=detectron2_eval
#SBATCH --output=logs/detectron2/detectron2_eval_%j.out
#SBATCH --error=logs/detectron2/detectron2_eval_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Create logs directory if it doesn't exist
mkdir -p logs/detectron2

# Change to project directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Load environment
source $HOME/.bashrc
conda activate detectron2

# Run the evaluation script with the config file
python scripts/detectron2/test/test_detectron2.py \
    --config configs/detectron2/detectron2_test_config.yaml

echo "Evaluation complete!" 