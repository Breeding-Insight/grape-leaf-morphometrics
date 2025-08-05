#!/bin/bash
#SBATCH --job-name=optuna_pointrend_optimization
#SBATCH --output=logs/detectron2_pointrend/optuna_optimization_%j.out
#SBATCH --error=logs/detectron2_pointrend/optuna_optimization_%j.err
#SBATCH --partition=regular
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# =============================================================================
# OPTUNA POINTREND HYPERPARAMETER OPTIMIZATION - Bayesian Search
# =============================================================================
#
# This job runs Bayesian hyperparameter optimization for PointRend using Optuna:
# - 30 trials of systematic hyperparameter exploration
# - 13 hyperparameters optimized including LR schedule, PointRend params, ROI sampling
# - Each trial trains for up to 8000 iterations with early stopping
# - SQLite database for persistence and resumability
# - Expected runtime: 60-120 hours depending on early stopping frequency
#
# Optimized parameters:
# - Core training: learning_rate, batch_size, weight_decay, warmup_iters, lr_decay_step_ratio
# - ROI sampling: roi_batch_size_per_image, roi_positive_fraction, roi_score_threshold  
# - PointRend: subdivision_steps, subdivision_num_points, importance_sample_ratio
# - Data augmentation: horizontal_flip_prob, vertical_flip_prob
#
# Expected outcome: 2-5% improvement in segmentation AP over manual tuning
#
# =============================================================================

# Print job information
echo "========================================="
echo "🔬 OPTUNA POINTREND OPTIMIZATION JOB"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Planned Trials: 130"
echo "Expected Runtime: 60-120 hours"
echo "========================================="

# Create logs directory if it doesn't exist
mkdir -p logs/detectron2_pointrend

# Set up environment
echo "🔧 Setting up environment..."
source ~/.bashrc

# Activate detectron2 conda environment
echo "🐍 Activating detectron2 environment..."
source activate detectron2

# Verify environment
echo "📋 Environment verification:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Check if detectron2, PointRend, and Optuna are available
echo "🔍 Checking required packages..."
python -c "
import detectron2
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.projects.point_rend.roi_heads import PointRendROIHeads
from detectron2.projects.point_rend.mask_head import PointRendMaskHead
import optuna
import yaml
print('✅ Detectron2, PointRend, and Optuna successfully imported')
print(f'Optuna version: {optuna.__version__}')
"

if [ $? -ne 0 ]; then
    echo "❌ Environment setup failed - exiting"
    echo "💡 Install missing packages with:"
    echo "   conda install -c conda-forge optuna"
    echo "   pip install optuna"
    exit 1
fi

# Set working directory
cd /workdir/data/grape/grape_pheno/grape-leaf-morphometrics

# Verify required files exist
echo "📁 Verifying required files..."
BASE_CONFIG="scripts/detectron2_v2/train_val/X101-FPN_pointrend/config_detectron2.yaml"
TRAIN_SCRIPT="scripts/detectron2_v2/train_val/X101-FPN_pointrend/train_detectron2.py"
OPTUNA_SCRIPT="scripts/detectron2_v2/train_val/X101-FPN_pointrend/optuna_detectron2_seg_search.py"

if [ ! -f "$BASE_CONFIG" ]; then
    echo "❌ Base config file not found: $BASE_CONFIG"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "❌ Training script not found: $TRAIN_SCRIPT"
    exit 1
fi

if [ ! -f "$OPTUNA_SCRIPT" ]; then
    echo "❌ Optuna optimization script not found: $OPTUNA_SCRIPT"
    exit 1
fi

echo "✅ All required files found"

# Display configuration summary
echo "📋 Optimization Configuration Summary:"
echo "Base config: $BASE_CONFIG"
echo "Training script: $TRAIN_SCRIPT"
echo "Optuna script: $OPTUNA_SCRIPT"
echo "Model architecture: ResNeXt-101 + PointRend"
echo "Output directory: optuna_pointrend_trials/"
echo "Study name: pointrend_peduncle_bayesian_opt"

# Print GPU information
echo "🖥️  GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Print dataset information
echo "📊 Dataset Information:"
if [ -f "data/annotations/coco/train/_annotations.coco.json" ]; then
    echo "✅ Training annotations found"
    TRAIN_SAMPLES=$(python -c "import json; data=json.load(open('data/annotations/coco/train/_annotations.coco.json')); print(len(data['images']))")
    echo "   Training samples: $TRAIN_SAMPLES"
else
    echo "❌ Training annotations not found"
fi

if [ -f "data/annotations/coco/valid/_annotations.coco.json" ]; then
    echo "✅ Validation annotations found"
    VAL_SAMPLES=$(python -c "import json; data=json.load(open('data/annotations/coco/valid/_annotations.coco.json')); print(len(data['images']))")
    echo "   Validation samples: $VAL_SAMPLES"
else
    echo "❌ Validation annotations not found"
fi

# Create output directory
mkdir -p optuna_pointrend_trials

# Display hyperparameter search space
echo "🎯 Hyperparameter Search Space:"
echo "   learning_rate: 1e-4 to 4e-4 (log scale)"
echo "   batch_size: 4 to 8"
echo "   weight_decay: 1e-5 to 1e-3 (log scale)"
echo "   warmup_iters: 500 to 2000"
echo "   lr_decay_step_ratio: 0.5 to 0.8"
echo "   roi_batch_size_per_image: 64 to 256"
echo "   roi_positive_fraction: 0.25 to 0.5"
echo "   roi_score_threshold: 0.65 to 0.8"
echo "   pointrend_subdivision_steps: 3 to 7"
echo "   pointrend_subdivision_num_points: 512 to 1024"
echo "   pointrend_importance_sample_ratio: 0.7 to 0.9"
echo "   horizontal_flip_prob: 0.3 to 0.7"
echo "   vertical_flip_prob: 0.1 to 0.5"

# Start optimization
echo "========================================="
echo "🚀 STARTING OPTUNA OPTIMIZATION"
echo "========================================="
echo "Start time: $(date)"
echo "Optimization strategy: TPE (Tree-structured Parzen Estimator)"
echo "Primary metric: segm/AP75 (maximize)"
echo "Trials planned: 130"

# Run the optimization with error handling
python "$OPTUNA_SCRIPT"
OPTIMIZATION_EXIT_CODE=$?

echo "========================================="
echo "📊 OPTIMIZATION COMPLETED"
echo "========================================="
echo "End time: $(date)"
echo "Exit code: $OPTIMIZATION_EXIT_CODE"

# Check if optimization was successful
if [ $OPTIMIZATION_EXIT_CODE -eq 0 ]; then
    echo "✅ Optimization completed successfully!"
    
    # Display results directory contents
    RESULTS_DIR="optuna_pointrend_trials"
    if [ -d "$RESULTS_DIR" ]; then
        echo "📁 Results directory overview:"
        echo "   Total trials conducted: $(ls -1 "$RESULTS_DIR"/trial_* 2>/dev/null | wc -l)"
        
        # Show optimization results if available
        if [ -d "$RESULTS_DIR/optimization_results" ]; then
            echo "📊 Optimization Results:"
            
            if [ -f "$RESULTS_DIR/optimization_results/best_trial.json" ]; then
                echo "🏆 Best Trial Results:"
                cat "$RESULTS_DIR/optimization_results/best_trial.json"
            fi
            
            if [ -f "$RESULTS_DIR/optimization_results/optimization_report.txt" ]; then
                echo "📋 Optimization Report (last 20 lines):"
                tail -20 "$RESULTS_DIR/optimization_results/optimization_report.txt"
            fi
        fi
        
        # Show SQLite database info
        if [ -f "$RESULTS_DIR/pointrend_peduncle_bayesian_opt.db" ]; then
            echo "✅ Optimization database saved: pointrend_peduncle_bayesian_opt.db"
            echo "   Database can be used to resume optimization or analyze results"
        fi
    fi
    
    # Display final GPU status
    echo "🖥️  Final GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    
    # Recommendations for next steps
    echo "🎯 Next Steps:"
    echo "   1. Review best_trial.json for optimal hyperparameters"
    echo "   2. Run final training with best parameters"
    echo "   3. Consider extending optimization with more trials if needed"
    echo "   4. Analyze parameter importance in optimization_report.txt"
    
else
    echo "❌ Optimization failed with exit code: $OPTIMIZATION_EXIT_CODE"
    echo "Check the error log for details: logs/detectron2_pointrend/optuna_optimization_${SLURM_JOB_ID}.err"
    
    # Show partial results if any trials completed
    if [ -d "optuna_pointrend_trials" ]; then
        COMPLETED_TRIALS=$(ls -1 optuna_pointrend_trials/trial_* 2>/dev/null | wc -l)
        if [ $COMPLETED_TRIALS -gt 0 ]; then
            echo "⚠️  Partial results available: $COMPLETED_TRIALS trials completed"
            echo "   Check optuna_pointrend_trials/ for partial optimization results"
        fi
    fi
fi

echo "========================================="
echo "🏁 OPTIMIZATION JOB COMPLETED"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Total runtime: $((SECONDS / 3600)) hours $((SECONDS % 3600 / 60)) minutes"
echo "End time: $(date)"

# Final environment cleanup
conda deactivate

exit $OPTIMIZATION_EXIT_CODE
