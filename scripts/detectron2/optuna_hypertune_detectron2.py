import os
import json
import subprocess
import optuna
import logging
from datetime import datetime
from pathlib import Path
import tempfile
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PointRendOptimizer:
    """
    Bayesian optimization for PointRend hyperparameters using Optuna.
    Based on experimental results and designed for production reliability.
    """
    
    def __init__(self, 
                 base_config_path="scripts/detectron2_v2/train_val/X101-FPN_pointrend/config_detectron2.yaml",
                 training_script="scripts/detectron2_v2/train_val/X101-FPN_pointrend/train_detectron2.py",
                 base_output_dir="optuna_pointrend_trials",
                 study_name="pointrend_peduncle_optimization",
                 n_trials=50):
        
        self.base_config_path = Path(base_config_path)
        self.training_script = Path(training_script)
        self.base_output_dir = Path(base_output_dir)
        self.study_name = study_name
        self.n_trials = n_trials
        
        # Ensure output directory exists
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Load base config for modification
        with open(self.base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
    
    def suggest_hyperparameters(self, trial):
        """
        Define hyperparameter search space based on your experimental results.
        Using informed priors from your grid search findings.
        """
        
        # Training hyperparameters - based on your lr_scaling_results
        # Your best result was batch_size=6, lr=0.0002
        params = {
            # Core training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 4e-4, log=True),
            'batch_size': trial.suggest_int('batch_size', 4, 8),
            
            # ROI Sampling - based on your roi_sampling_results
            # Best was: batch_size_per_image=64, positive_fraction=0.5, score_threshold=0.75
            'roi_batch_size_per_image': trial.suggest_int('roi_batch_size_per_image', 64, 256),
            'roi_positive_fraction': trial.suggest_float('roi_positive_fraction', 0.25, 0.5),
            'roi_score_threshold': trial.suggest_float('roi_score_threshold', 0.65, 0.8),
            
            # PointRend Architecture Parameters - based on your pointrend experiments
            'pointrend_subdivision_steps': trial.suggest_int('pointrend_subdivision_steps', 3, 7),
            'pointrend_subdivision_num_points': trial.suggest_int('pointrend_subdivision_num_points', 512, 1024),
            'pointrend_importance_sample_ratio': trial.suggest_float('pointrend_importance_sample_ratio', 0.7, 0.9),
            
            # Additional optimization parameters
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'warmup_iters': trial.suggest_int('warmup_iters', 500, 2000),
            
            # Learning rate schedule optimization
            'lr_decay_step_ratio': trial.suggest_float('lr_decay_step_ratio', 0.5, 0.8),
            
            # Augmentation parameters
            'horizontal_flip_prob': trial.suggest_float('horizontal_flip_prob', 0.3, 0.7),
            'vertical_flip_prob': trial.suggest_float('vertical_flip_prob', 0.1, 0.5),
        }
        
        return params
    
    def create_config_for_trial(self, trial_params, trial_number):
        """
        Create a modified config file for this specific trial.
        Modify the config systematically based on the config structure.
        """
        
        # Deep copy base config
        import copy
        trial_config = copy.deepcopy(self.base_config)
        
        # Update solver parameters (based on your config structure)
        if 'solver' not in trial_config:
            trial_config['solver'] = {}
        
        trial_config['solver']['base_lr'] = trial_params['learning_rate']
        trial_config['solver']['ims_per_batch'] = trial_params['batch_size']
        trial_config['solver']['weight_decay'] = trial_params.get('weight_decay', 1e-4)
        trial_config['solver']['warmup_iters'] = trial_params.get('warmup_iters', 1000)
        
        # Calculate and set LR decay step based on ratio
        max_iter = trial_config['solver']['max_iter']
        decay_step = int(max_iter * trial_params.get('lr_decay_step_ratio', 0.75))
        trial_config['solver']['steps'] = [decay_step]
        
        # Update ROI heads parameters (based on your config structure)
        if 'roi_heads' not in trial_config:
            trial_config['roi_heads'] = {}
        
        trial_config['roi_heads']['batch_size_per_image'] = trial_params['roi_batch_size_per_image']
        trial_config['roi_heads']['positive_fraction'] = trial_params['roi_positive_fraction']
        trial_config['roi_heads']['SCORE_THRESH_TEST'] = trial_params['roi_score_threshold']
        
        # Update PointRend parameters (FIXED: correct nested structure)
        if 'model' not in trial_config:
            trial_config['model'] = {}
        if 'pointrend' not in trial_config['model']:
            trial_config['model']['pointrend'] = {}
        
        trial_config['model']['pointrend']['subdivision_steps'] = trial_params['pointrend_subdivision_steps']
        trial_config['model']['pointrend']['subdivision_num_points'] = trial_params['pointrend_subdivision_num_points']
        trial_config['model']['pointrend']['importance_sample_ratio'] = trial_params['pointrend_importance_sample_ratio']
        
        # Update input/augmentation parameters
        if 'INPUT' not in trial_config:
            trial_config['INPUT'] = {}
        
        trial_config['INPUT']['HORIZONTAL_FLIP_PROB'] = trial_params.get('horizontal_flip_prob', 0.5)
        trial_config['INPUT']['VERTICAL_FLIP_PROB'] = trial_params.get('vertical_flip_prob', 0.2)
        
        # Create trial-specific config file
        trial_config_path = self.base_output_dir / f"trial_{trial_number}_config.yaml"
        with open(trial_config_path, 'w') as f:
            yaml.dump(trial_config, f, default_flow_style=False)
        
        return trial_config_path
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        """
        
        # Suggest hyperparameters for this trial
        trial_params = self.suggest_hyperparameters(trial)
        
        # Create output directory for this trial
        trial_dir = self.base_output_dir / f"trial_{trial.number}"
        trial_dir.mkdir(exist_ok=True)
        
        # Create trial-specific config
        trial_config_path = self.create_config_for_trial(trial_params, trial.number)
        
        # Log trial parameters
        logger.info(f"Trial {trial.number} parameters: {trial_params}")
        
        # Save trial parameters for debugging
        with open(trial_dir / "trial_params.json", 'w') as f:
            json.dump(trial_params, f, indent=2)
        
        # Build command to run training script (matches your actual script interface)
        command = [
            "python", str(self.training_script),
            "--config", str(trial_config_path),
            "--output_dir", str(trial_dir),
        ]
        
        # Add resume flag if checkpoint exists
        if (trial_dir / "last_checkpoint").exists():
            command.append("--resume")
        
        try:
            # Run training with timeout protection
            logger.info(f"Starting trial {trial.number} training...")
            result = subprocess.run(
                command, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=3600 * 4.5,  # 4.5 hour timeout (aligned with 7000 iter max, generous buffer)
                cwd=self.base_config_path.parent.parent.parent.parent.parent  # Set working directory to project root
            )
            
            # Log training output for debugging
            with open(trial_dir / "training_log.txt", 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Trial {trial.number} timed out")
            return 0.0
        except subprocess.CalledProcessError as e:
            logger.error(f"Trial {trial.number} failed with return code {e.returncode}")
            
            # Log stdout and stderr for detailed diagnostics
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")

            # Save the complete error log to the trial directory for inspection
            error_log_path = trial_dir / f"trial_{trial.number}_error.log"
            with open(error_log_path, 'w') as f:
                f.write(f"--- STDOUT ---\n{e.stdout}\n")
                f.write(f"--- STDERR ---\n{e.stderr}\n")
            logger.error(f"Full error log saved to: {error_log_path}")
            
            return 0.0
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with exception: {e}")
            return 0.0
        
        # Read metrics from results
        # Look for common Detectron2 output files
        metrics_files = [
            trial_dir / "metrics.json",
            trial_dir / "last_checkpoint",
            trial_dir / "model_final.pth"
        ]
        
        # Try to find and parse metrics
        primary_metric = self.extract_primary_metric(trial_dir)
        
        if primary_metric is None:
            logger.warning(f"Trial {trial.number}: Could not extract primary metric")
            return 0.0
        
        logger.info(f"Trial {trial.number} completed with metric: {primary_metric}")
        return primary_metric
    
    def extract_primary_metric(self, trial_dir):
        """
        Extract primary metric from training results.
        Based on your actual training output structure.
        """
        
        # Look for early_stopping_summary.txt (from your actual output logs)
        early_stopping_file = trial_dir / "early_stopping_summary.txt"
        if early_stopping_file.exists():
            try:
                with open(early_stopping_file, 'r') as f:
                    content = f.read()
                
                # Parse the early stopping summary for best segm/AP75
                # Based on your log format: "Best segm/AP75: 73.407190"
                import re
                ap75_match = re.search(r'Best segm/AP75:\s*([\d.]+)', content)
                if ap75_match:
                    return float(ap75_match.group(1))
            except Exception as e:
                logger.warning(f"Could not parse early_stopping_summary.txt: {e}")
        
        # Look for final evaluation output in log files
        log_files = list(trial_dir.glob("*.log"))
        if log_files:
            try:
                with open(log_files[0], 'r') as f:
                    content = f.read()
                
                # Look for final validation metrics
                # This would depend on your exact log format
                import re
                ap_match = re.search(r'segm/AP75.*?:\s*([\d.]+)', content)
                if ap_match:
                    return float(ap_match.group(1))
            except Exception as e:
                logger.warning(f"Could not parse log file: {e}")
        
        # Look for metrics.json if your script outputs this
        metrics_path = trial_dir / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                # Try different possible metric names
                for metric_name in ["segm/AP75", "early_stopping/segm/AP75", "AP75", "best_ap75"]:
                    if metric_name in metrics:
                        return float(metrics[metric_name])
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse metrics.json: {e}")
        
        return None
    
    def run_optimization(self):
        """
        Run the complete Bayesian optimization study.
        """
        
        logger.info(f"Starting Bayesian optimization study: {self.study_name}")
        logger.info(f"Planned trials: {self.n_trials}")
        
        # Create study with SQLite storage for persistence
        storage_url = f"sqlite:///{self.base_output_dir}/{self.study_name}.db"
        
        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",  # Maximize AP75
            sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible results
            pruner=optuna.pruners.MedianPruner(  # Aggressive early stopping for poor trials (48h optimization)
                n_startup_trials=3,
                n_warmup_steps=5,
                interval_steps=3
            )
        )
        
        # Run optimization
        try:
            study.optimize(self.objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        
        # Save results
        self.save_results(study)
        
        return study
    
    def save_results(self, study):
        """
        Save optimization results with comprehensive analysis.
        """
        
        results_dir = self.base_output_dir / "optimization_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save best trial
        best_trial_path = results_dir / "best_trial.json"
        with open(best_trial_path, 'w') as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_trial_number": study.best_trial.number,
                "n_trials": len(study.trials)
            }, f, indent=2)
        
        # Save all trials
        all_trials_path = results_dir / "all_trials.json"
        trials_data = []
        for trial in study.trials:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None
            })
        
        with open(all_trials_path, 'w') as f:
            json.dump(trials_data, f, indent=2)
        
        # Generate optimization report
        self.generate_optimization_report(study, results_dir)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
    
    def generate_optimization_report(self, study, results_dir):
        """
        Generate comprehensive optimization report.
        """
        
        report_path = results_dir / "optimization_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("POINTREND BAYESIAN OPTIMIZATION RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"Study Name: {self.study_name}\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"Successful Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Optimization Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Primary Metric: segm/AP75 (maximize)\n\n")
            
            f.write("BEST PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Trial Number: {study.best_trial.number}\n")
            f.write(f"Best Value: {study.best_value:.6f}\n")
            f.write("Best Parameters:\n")
            for param, value in study.best_params.items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
            
            # Parameter importance analysis
            if len(study.trials) > 10:  # Need sufficient trials for importance analysis
                try:
                    importance = optuna.importance.get_param_importances(study)
                    f.write("PARAMETER IMPORTANCE:\n")
                    f.write("-" * 40 + "\n")
                    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {param}: {imp:.4f}\n")
                    f.write("\n")
                except Exception as e:
                    f.write(f"Could not compute parameter importance: {e}\n\n")
            
            # Trial summary
            f.write("TRIAL SUMMARY:\n")
            f.write("-" * 40 + "\n")
            successful_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if successful_trials:
                values = [t.value for t in successful_trials]
                f.write(f"Best Value: {max(values):.6f}\n")
                f.write(f"Mean Value: {sum(values)/len(values):.6f}\n")
                f.write(f"Std Value: {(sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5:.6f}\n")
        
        logger.info(f"Optimization report saved to: {report_path}")


def main():
    """
    Main execution function with proper error handling.
    """
    
    # Initialize optimizer with specific paths
    optimizer = PointRendOptimizer(
        base_config_path="scripts/detectron2_v2/train_val/X101-FPN_pointrend/config_detectron2.yaml",
        training_script="scripts/detectron2_v2/train_val/X101-FPN_pointrend/train_detectron2.py",
        study_name="pointrend_peduncle_bayesian_opt",
        n_trials=15  # Optimized for 48-hour completion with sufficient iterations to reach peak performance
    )
    
    # Run optimization
    study = optimizer.run_optimization()
    
    print(f"\n{'='*60}")
    print("BAYESIAN OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best AP75: {study.best_value:.6f}")
    print(f"Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
