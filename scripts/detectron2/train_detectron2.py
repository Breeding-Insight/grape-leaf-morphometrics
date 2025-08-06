import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg, CfgNode as CN
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.datasets.coco import load_coco_json
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.events import get_event_storage
from detectron2.projects import point_rend
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import threading
import time
import subprocess
import logging
import json
import shutil
import copy
import torch
import gc
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EarlyStoppingHook(HookBase):
    """
    Early stopping hook for Detectron2 training with segmentation AP monitoring.
    
    Monitors validation metrics and stops training when performance plateaus,
    preventing overfitting and saving computational resources.
    """
    
    def __init__(self, cfg, val_dataset_name: str, early_stopping_config: Dict):
        """
        Initialize early stopping hook.
        
        Args:
            cfg: Detectron2 config
            val_dataset_name: Name of validation dataset
            early_stopping_config: Early stopping configuration from YAML
        """
        self.cfg = cfg.clone()
        self.val_dataset_name = val_dataset_name
        
        # Early stopping parameters
        self.enabled = early_stopping_config.get("enabled", True)
        self.metric_name = early_stopping_config.get("metric", "segm/AP")
        self.patience = early_stopping_config.get("patience", 4)
        self.min_delta = early_stopping_config.get("min_delta", 0.001)
        self.mode = early_stopping_config.get("mode", "max")
        self.eval_period = early_stopping_config.get("eval_period", 500)
        self.min_iterations = early_stopping_config.get("min_iterations", 2000)
        self.restore_best_weights = early_stopping_config.get("restore_best_weights", True)
        self.save_best_model = early_stopping_config.get("save_best_model", True)
        self.best_model_filename = early_stopping_config.get("best_model_filename", "model_best.pth")
        self.verbose = early_stopping_config.get("verbose", True)
        
        # Internal state tracking
        self.best_metric = float('-inf') if self.mode == 'max' else float('inf')
        self.best_iteration = 0
        self.patience_counter = 0
        self.should_stop = False
        self.best_checkpoint_path = None
        
        # Evaluation components
        self.evaluator = None
        self.val_loader = None
        
        # Logging and debugging
        self.evaluation_history = []
        self.early_stopping_log = []
        
        logger.info(f"Early Stopping Hook initialized:")
        logger.info(f"  Metric: {self.metric_name}")
        logger.info(f"  Patience: {self.patience}")
        logger.info(f"  Min Delta: {self.min_delta}")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Eval Period: {self.eval_period}")
        logger.info(f"  Min Iterations: {self.min_iterations}")
    
    def before_train(self):
        """Initialize evaluation components before training starts."""
        if not self.enabled:
            logger.info("Early stopping disabled in configuration")
            return
            
        try:
            # Initialize evaluator and data loader
            self.evaluator = COCOEvaluator(
                self.val_dataset_name, 
                self.cfg, 
                False, 
                output_dir=os.path.join(self.cfg.OUTPUT_DIR, "early_stopping_eval")
            )
            self.val_loader = build_detection_test_loader(self.cfg, self.val_dataset_name)
            
            # Create directory for best model if saving enabled
            if self.save_best_model:
                self.best_checkpoint_path = os.path.join(self.cfg.OUTPUT_DIR, self.best_model_filename)
            
            logger.info("Early stopping evaluation components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize early stopping components: {str(e)}")
            logger.warning("Disabling early stopping due to initialization failure")
            self.enabled = False
    
    def after_step(self):
        """Check for early stopping after each training step."""
        if not self.enabled:
            return
            
        current_iter = self.trainer.iter
        
        # Only evaluate at specified intervals and after minimum iterations
        if (current_iter % self.eval_period == 0 and 
            current_iter >= self.min_iterations and 
            current_iter > 0):
            
            logger.info(f"Running early stopping evaluation at iteration {current_iter}")
            self._evaluate_and_check_stopping(current_iter)
            
            # Stop training if criteria met
            if self.should_stop:
                logger.info(f"Early stopping triggered at iteration {current_iter}")
                self._handle_early_stopping()
                # Set trainer to stop
                self.trainer.storage.put_scalar("early_stopping/triggered", 1)
                # This will be caught by the training loop
                raise StopTraining(f"Early stopping at iteration {current_iter}")
    
    def _evaluate_and_check_stopping(self, current_iter: int):
        """
        Run evaluation and check early stopping criteria.
        
        Args:
            current_iter: Current training iteration
        """
        try:
            # Run evaluation
            logger.info("Running validation evaluation...")
            
            # Ensure model is in eval mode for evaluation
            was_training = self.trainer.model.training
            self.trainer.model.eval()
            
            eval_results = inference_on_dataset(
                self.trainer.model, 
                self.val_loader, 
                self.evaluator
            )
            
            # CRITICAL: Restore training mode
            if was_training:
                self.trainer.model.train()
            
            # Extract the metric we're monitoring
            current_metric = self._extract_metric(eval_results, self.metric_name)
            
            if current_metric is None:
                logger.warning(f"Could not extract metric '{self.metric_name}' from evaluation results")
                logger.warning(f"Available metrics: {list(eval_results.keys())}")
                return
            
            # Log current performance
            logger.info(f"Iteration {current_iter}: {self.metric_name} = {current_metric:.6f}")
            
            # Store evaluation history
            eval_record = {
                'iteration': current_iter,
                'metric_value': current_metric,
                'metric_name': self.metric_name
            }
            self.evaluation_history.append(eval_record)
            
            # Log to tensorboard/events
            storage = get_event_storage()
            storage.put_scalar(f"early_stopping/{self.metric_name}", current_metric)
            storage.put_scalar("early_stopping/patience_counter", self.patience_counter)
            storage.put_scalar("early_stopping/best_metric", self.best_metric)
            
            # Check if this is the best performance so far
            is_improvement = self._is_improvement(current_metric, self.best_metric)
            
            if is_improvement:
                improvement = abs(current_metric - self.best_metric)
                logger.info(f"✓ New best {self.metric_name}: {current_metric:.6f} "
                           f"(improvement: +{improvement:.6f})")
                
                # Update best metrics
                self.best_metric = current_metric
                self.best_iteration = current_iter
                self.patience_counter = 0
                
                # Save best model if enabled
                if self.save_best_model:
                    self._save_best_checkpoint(current_iter)
                
                # Log improvement
                early_stopping_log = {
                    'iteration': current_iter,
                    'event': 'improvement',
                    'metric_value': current_metric,
                    'improvement': improvement,
                    'patience_counter': self.patience_counter
                }
                self.early_stopping_log.append(early_stopping_log)
                
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
                
                # Log no improvement
                early_stopping_log = {
                    'iteration': current_iter,
                    'event': 'no_improvement', 
                    'metric_value': current_metric,
                    'best_metric': self.best_metric,
                    'patience_counter': self.patience_counter
                }
                self.early_stopping_log.append(early_stopping_log)
                
                # Check if we should stop
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping criteria met:")
                    logger.info(f"  No improvement for {self.patience} evaluations")
                    logger.info(f"  Best {self.metric_name}: {self.best_metric:.6f} at iteration {self.best_iteration}")
                    logger.info(f"  Current {self.metric_name}: {current_metric:.6f}")
                    
                    self.should_stop = True
            
            # Save evaluation history periodically
            self._save_evaluation_history()
            
        except Exception as e:
            logger.error(f"Error during early stopping evaluation: {str(e)}")
            logger.warning("Continuing training despite evaluation error")
            
            # Ensure model is restored to training mode even after error
            if hasattr(self, 'trainer') and self.trainer and hasattr(self.trainer, 'model'):
                self.trainer.model.train()
                logger.info("Model restored to training mode after evaluation error")
    
    def _extract_metric(self, eval_results: Dict, metric_name: str) -> Optional[float]:
        """
        Extract specific metric from evaluation results.
        
        Args:
            eval_results: Results from COCO evaluation
            metric_name: Name of metric to extract
            
        Returns:
            Metric value or None if not found
        """
        try:
            # Handle nested metric names (e.g., "segm/AP")
            if "/" in metric_name:
                category, metric = metric_name.split("/", 1)
                if category in eval_results and metric in eval_results[category]:
                    return float(eval_results[category][metric])
            
            # Handle direct metric names
            if metric_name in eval_results:
                return float(eval_results[metric_name])
            
            # Special handling for common metric aliases (only non-self-referencing ones)
            metric_aliases = {
                "AP": "segm/AP"  # Default to segmentation AP
            }
            
            if metric_name in metric_aliases:
                alias = metric_aliases[metric_name]
                return self._extract_metric(eval_results, alias)
            
            return None
            
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Failed to extract metric '{metric_name}': {str(e)}")
            return None
    
    def _is_improvement(self, current: float, best: float) -> bool:
        """
        Check if current metric represents an improvement.
        
        Args:
            current: Current metric value
            best: Best metric value so far
            
        Returns:
            True if current is better than best by at least min_delta
        """
        if self.mode == 'max':
            return current > (best + self.min_delta)
        else:  # mode == 'min'
            return current < (best - self.min_delta)
    
    def _save_best_checkpoint(self, iteration: int):
        """
        Save checkpoint for best performing model.
        
        Args:
            iteration: Current training iteration
        """
        try:
            # Save model state
            checkpointer = DetectionCheckpointer(
                self.trainer.model, 
                save_dir=self.cfg.OUTPUT_DIR
            )
            
            # Save with best model filename
            checkpointer.save(self.best_model_filename.replace('.pth', ''))
            
            logger.info(f"Saved best model checkpoint: {self.best_checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save best model checkpoint: {str(e)}")
    
    def _save_evaluation_history(self):
        """Save evaluation history to JSON file for analysis."""
        try:
            history_file = os.path.join(self.cfg.OUTPUT_DIR, "early_stopping_history.json")
            
            history_data = {
                'evaluation_history': self.evaluation_history,
                'early_stopping_log': self.early_stopping_log,
                'config': {
                    'metric_name': self.metric_name,
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                    'mode': self.mode,
                    'eval_period': self.eval_period
                },
                'best_performance': {
                    'best_metric': self.best_metric,
                    'best_iteration': self.best_iteration
                }
            }
            
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save evaluation history: {str(e)}")
    
    def _handle_early_stopping(self):
        """Handle early stopping procedures."""
        try:
            logger.info("=== Early Stopping Triggered ===")
            logger.info(f"Training stopped at iteration: {self.trainer.iter}")
            logger.info(f"Best {self.metric_name}: {self.best_metric:.6f} at iteration {self.best_iteration}")
            
            # CRITICAL: Preserve the actual best metric values before checkpoint restoration
            # This prevents the checkpoint loading from overwriting our current best values
            actual_best_metric = self.best_metric
            actual_best_iteration = self.best_iteration
            
            # Restore best weights if enabled
            if self.restore_best_weights and self.best_checkpoint_path:
                logger.info("Restoring best model weights...")
                try:
                    checkpointer = DetectionCheckpointer(self.trainer.model)
                    checkpointer.load(self.best_checkpoint_path.replace('.pth', ''))
                    logger.info("Best model weights restored successfully")
                except Exception as e:
                    logger.error(f"Failed to restore best weights: {str(e)}")
            
            # CRITICAL: Restore the actual best values after checkpoint loading
            # The checkpoint loading may have overwritten these with stale values
            self.best_metric = actual_best_metric
            self.best_iteration = actual_best_iteration
            
            # Save final evaluation history
            self._save_evaluation_history()
            
            # Create early stopping summary
            self._create_stopping_summary()
            
        except Exception as e:
            logger.error(f"Error handling early stopping: {str(e)}")
    
    def _create_stopping_summary(self):
        """Create a summary of early stopping results."""
        try:
            summary_file = os.path.join(self.cfg.OUTPUT_DIR, "early_stopping_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write("=== Early Stopping Summary ===\n\n")
                f.write(f"Training stopped at iteration: {self.trainer.iter}\n")
                f.write(f"Best {self.metric_name}: {self.best_metric:.6f}\n")
                f.write(f"Best iteration: {self.best_iteration}\n")
                f.write(f"Patience used: {self.patience_counter}/{self.patience}\n")
                f.write(f"Total evaluations: {len(self.evaluation_history)}\n\n")
                
                f.write("Configuration:\n")
                f.write(f"  Metric: {self.metric_name}\n")
                f.write(f"  Patience: {self.patience}\n")
                f.write(f"  Min Delta: {self.min_delta}\n")
                f.write(f"  Mode: {self.mode}\n")
                f.write(f"  Eval Period: {self.eval_period}\n\n")
                
                if self.evaluation_history:
                    f.write("Recent evaluation history:\n")
                    for eval_record in self.evaluation_history[-5:]:
                        f.write(f"  Iter {eval_record['iteration']}: {eval_record['metric_value']:.6f}\n")
            
            logger.info(f"Early stopping summary saved: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to create stopping summary: {str(e)}")


class StopTraining(Exception):
    """Exception to signal early stopping to training loop."""
    pass


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass


def print_pointrend_installation_guide():
    """Print comprehensive PointRend installation instructions."""
    logger.info("=" * 80)
    logger.info("🔧 POINTREND INSTALLATION GUIDE")
    logger.info("=" * 80)
    logger.info("")
    logger.info("PointRend is not installed or not properly configured. To fix this:")
    logger.info("")
    logger.info("📥 OPTION 1: Install from source (recommended)")
    logger.info("   1. Clone detectron2 with projects:")
    logger.info("      git clone https://github.com/facebookresearch/detectron2.git")
    logger.info("   2. Install PointRend project:")
    logger.info("      cd detectron2/projects/PointRend")
    logger.info("      pip install -e .")
    logger.info("")
    logger.info("📦 OPTION 2: Install detectron2 with all projects")
    logger.info("   pip install 'git+https://github.com/facebookresearch/detectron2.git#subdirectory=projects/PointRend'")
    logger.info("")
    logger.info("🔍 OPTION 3: Verify existing installation")
    logger.info("   Check if PointRend modules can be imported:")
    logger.info("   python -c \"from detectron2.projects.point_rend import add_pointrend_config; print('PointRend OK')\"")
    logger.info("")
    logger.info("🎛️  OPTION 4: Use standard Mask R-CNN instead")
    logger.info("   Set model.pointrend.enabled: false in your config")
    logger.info("")
    logger.info("=" * 80)


class RobustDatasetMapper(DatasetMapper):
    """Enhanced dataset mapper with better error handling and logging."""
    
    def __init__(self, cfg, is_train=True, augmentations=None):
        # DatasetMapper.from_config() signature changed - use from_config method
        if augmentations is not None:
            # Create a temporary config with augmentations
            import copy
            temp_cfg = copy.deepcopy(cfg)
            temp_cfg.INPUT.AUGMENTATIONS = augmentations
            super().__init__(temp_cfg, is_train)
        else:
            super().__init__(cfg, is_train)
        self.failed_files = []
        
    def __call__(self, dataset_dict):
        try:
            return super().__call__(dataset_dict)
        except FileNotFoundError as e:
            self.failed_files.append(dataset_dict.get('file_name', 'Unknown'))
            logger.warning(f"Skipping missing file: {dataset_dict.get('file_name', 'Unknown')}")
            return None
        except Exception as e:
            logger.error(f"Error processing {dataset_dict.get('file_name', 'Unknown')}: {str(e)}")
            return None
    
    def get_failed_files(self) -> List[str]:
        """Return list of files that failed to load."""
        return self.failed_files.copy()


class DetectronConfigManager:
    """Centralized configuration management for Detectron2 training."""
    
    # Define supported model configurations
    # NOTE: PointRend configs are NOT in model zoo - they're in projects/PointRend/configs/
    MODEL_CONFIG_MATRIX = {
        ("resnet50", False): {
            "config_file": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            "weights_url": None,  # Use default from config
            "is_project_config": False
        },
        ("resnet50", True): {
            "config_file": "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml",
            # Use ImageNet-only backbone weights to avoid shape-mismatch with new PointRend heads
            "weights_url": "detectron2://ImageNetPretrained/FAIR/R-50.pkl",
            "is_project_config": True
        },
        ("resnext101", False): {
            "config_file": "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml",
            "weights_url": None,  # Use default from config
            "is_project_config": False
        },
        ("resnext101", True): {
            "config_file": "projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco.yaml",
            # Use the full PointRend pretrained model
            "weights_url": "https://dl.fbaipublicfiles.com/detectron2/PointRend/InstanceSegmentation/pointrend_rcnn_X_101_32x8d_FPN_3x_coco/28119989/model_final_ba17b9.pkl",
            "is_project_config": True
        }
    }
    
    def __init__(self, config_data: Dict, project_root: Path):
        self.config_data = config_data
        self.project_root = project_root
        self.cfg = None
        
    def build_config(self) -> CN:
        """Build complete Detectron2 configuration with proper precedence."""
        logger.info("Building Detectron2 configuration...")
        
        # Step 1: Initialize configuration with proper extensions
        self.cfg = get_cfg()
        
        # Step 2: Add PointRend support if needed (extends schema before loading base config)
        use_pointrend = self._should_use_pointrend()
        if use_pointrend:
            logger.info("Adding PointRend configuration support...")
            try:
                add_pointrend_config(self.cfg)
                logger.info("✅ PointRend configuration schema added successfully")
            except Exception as e:
                logger.error(f"❌ Failed to add PointRend configuration: {str(e)}")
                logger.error("PointRend is not properly installed. To fix this:")
                logger.error("  1. Install PointRend: cd detectron2/projects/PointRend && pip install -e .")
                logger.error("  2. Or disable PointRend in your config: model.pointrend.enabled: false")
                logger.error("  3. Or use a standard mask head config instead")
                
                # Automatically fall back to non-PointRend mode
                logger.warning("🔄 Automatically falling back to standard mask head training...")
                use_pointrend = False
                
                # Update config to reflect the fallback
                pointrend_config = self.config_data.get("model", {}).get("pointrend", {})
                pointrend_config["enabled"] = False
        
        # Step 3: Load the correct base configuration
        base_config_info = self._select_base_config()
        logger.info(f"Loading base configuration: {base_config_info['config_file']}")
        
        # Handle project configs differently from model zoo configs
        if base_config_info.get("is_project_config", False):
            # For project configs, we need to load from the local detectron2 installation
            self._load_project_config(base_config_info["config_file"])
        else:
            # Standard model zoo config
            self.cfg.merge_from_file(model_zoo.get_config_file(base_config_info["config_file"]))
        
        # Step 4: Set model weights - always use direct URLs for PointRend
        if base_config_info["weights_url"]:
            if base_config_info.get("is_project_config", False):
                # For project configs, always use direct URL (no model zoo mapping)
                self.cfg.MODEL.WEIGHTS = base_config_info["weights_url"]
                logger.info(f"Using direct weights URL for project config: {base_config_info['weights_url']}")
            else:
                # Standard model zoo weight handling
                try:
                    weights_url = model_zoo.get_checkpoint_url(base_config_info["weights_url"])
                    self.cfg.MODEL.WEIGHTS = weights_url
                    logger.info(f"Using model zoo weights: {base_config_info['weights_url']}")
                except Exception as e:
                    logger.warning(f"Failed to get model zoo weights: {str(e)}")
                    logger.info("Using random initialization - no pretrained weights")
                    self.cfg.MODEL.WEIGHTS = ""
        else:
            # Use default weights from the base config (for non-PointRend configs)
            try:
                self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base_config_info["config_file"])
                logger.info("Using default model weights from base configuration")
            except Exception as e:
                logger.warning(f"Failed to get default weights: {str(e)}")
                logger.info("Using random initialization - no pretrained weights")
                self.cfg.MODEL.WEIGHTS = ""
        
        # Step 5: Apply all custom configurations in correct order
        self._apply_custom_configurations()
        
        # Step 6: Apply PointRend-specific configurations if enabled
        if use_pointrend:
            self._apply_pointrend_configurations()
        
        # Step 7: Validate final configuration
        self._validate_configuration()
        
        logger.info("Configuration build completed successfully")
        return self.cfg
    
    # [Include all the existing methods from DetectronConfigManager - _should_use_pointrend, _select_base_config, etc.]
    # For brevity, I'll include the key methods for early stopping integration
    
    def _check_pointrend_availability(self) -> Tuple[bool, str]:
        """
        Check if PointRend is properly installed and available.
        
        Returns:
            Tuple of (is_available, error_message)
        """
        try:
            # Try importing core PointRend modules
            from detectron2.projects.point_rend import add_pointrend_config
            from detectron2.projects.point_rend.roi_heads import PointRendROIHeads
            from detectron2.projects.point_rend.mask_head import PointRendMaskHead
            from detectron2.projects.point_rend.point_features import point_sample
            return True, ""
        except ImportError as e:
            error_msg = f"PointRend modules not available: {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"PointRend check failed: {str(e)}"
            return False, error_msg

    def _should_use_pointrend(self) -> bool:
        """Determine if PointRend should be enabled."""
        requested = self.config_data.get("model", {}).get("pointrend", {}).get("enabled", False)
        
        if not requested:
            return False
            
        # Check if PointRend is actually available
        is_available, error_msg = self._check_pointrend_availability()
        
        if not is_available:
            logger.error(f"❌ PointRend not available: {error_msg}")
            logger.error("PointRend is not available. Falling back to regular mask head hypertuning.")
            
            # Show installation guide
            print_pointrend_installation_guide()
            
            # Automatically disable PointRend in config to prevent further errors
            pointrend_config = self.config_data.get("model", {}).get("pointrend", {})
            pointrend_config["enabled"] = False
            
            return False
        
        logger.info("✅ PointRend availability verified")
        return True
    
    def _select_base_config(self) -> Dict[str, str]:
        """Select appropriate base configuration based on requirements."""
        backbone = self.config_data.get("model", {}).get("backbone", "resnet50").lower()
        use_pointrend = self._should_use_pointrend()
        
        # Validate backbone type
        if backbone not in ["resnet50", "resnext101"]:
            raise ConfigurationError(f"Unsupported backbone: {backbone}. Supported: resnet50, resnext101")
        
        config_key = (backbone, use_pointrend)
        if config_key not in self.MODEL_CONFIG_MATRIX:
            raise ConfigurationError(f"No configuration available for {backbone} with PointRend={use_pointrend}")
        
        return self.MODEL_CONFIG_MATRIX[config_key]
    
    def _load_project_config(self, config_path: str):
        """Load PointRend configuration using working method without broken dependencies."""
        try:
            # Check if PointRend is available before proceeding
            is_available, error_msg = self._check_pointrend_availability()
            if not is_available:
                logger.error(f"❌ Cannot load PointRend config: {error_msg}")
                raise ConfigurationError(f"PointRend not available: {error_msg}")
            
            logger.info(f"✅ Loading PointRend config using manual configuration method")
            
            # Instead of loading the problematic config files with broken dependencies,
            # we'll build the PointRend configuration manually based on a working base config
            from detectron2 import model_zoo
            
            # Determine the appropriate base config based on the requested PointRend config
            if "pointrend_rcnn_R_50" in config_path:
                base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                logger.info(f"Using ResNet-50 base for PointRend")
            elif "pointrend_rcnn_X_101" in config_path:
                base_config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
                logger.info(f"Using ResNeXt-101 base for PointRend")
            elif "pointrend_rcnn_R_101" in config_path:
                base_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
                logger.info(f"Using ResNet-101 base for PointRend")
            else:
                # Default fallback
                base_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                logger.warning(f"Unknown PointRend config, using ResNet-50 base")
            
            # Load the working base config from model zoo
            self.cfg.merge_from_file(model_zoo.get_config_file(base_config))
            logger.info(f"✅ Loaded base config: {base_config}")
            
            # Apply PointRend-specific modifications manually
            self.cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
            self.cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
            self.cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""  # No RoI pooling for PointRend
            self.cfg.MODEL.ROI_MASK_HEAD.FC_DIM = 1024
            self.cfg.MODEL.ROI_MASK_HEAD.NUM_FC = 2
            self.cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True  # Enable point head
            
            # Pre-configure Point Head to ensure it exists (will be refined in _apply_pointrend_configurations)
            self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.cfg.MODEL.POINT_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
            self.cfg.MODEL.POINT_HEAD.NUM_CONV = 3
            self.cfg.MODEL.POINT_HEAD.CONV_DIM = 256
            self.cfg.MODEL.POINT_HEAD.NUM_POINTS = 196
            self.cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3.0
            self.cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
            self.cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
            self.cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 784
            
            logger.info(f"✅ PointRend configuration applied successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load PointRend config: {str(e)}")
            logger.warning("🔄 Falling back to equivalent standard Mask R-CNN config...")
            
            # Automatic fallback to standard config
            if "pointrend_rcnn_R_50" in config_path:
                fallback_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            elif "pointrend_rcnn_X_101" in config_path:
                fallback_config = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            elif "pointrend_rcnn_R_101" in config_path:
                fallback_config = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            else:
                fallback_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            
            logger.warning(f"🔄 Using fallback config: {fallback_config}")
            self.cfg.merge_from_file(model_zoo.get_config_file(fallback_config))
            
            # Disable PointRend in the config since we're using fallback
            pointrend_config = self.config_data.get("model", {}).get("pointrend", {})
            pointrend_config["enabled"] = False
            
            raise ConfigurationError(f"PointRend config loading failed, fell back to standard Mask R-CNN")
    
    def _apply_custom_configurations(self):
        """Apply all custom configurations in proper order."""
        logger.info("Applying custom configurations...")
        
        # Core configurations
        self._apply_dataset_config()
        self._apply_dataloader_config()
        self._apply_solver_config()
        self._apply_model_config()
        self._apply_roi_heads_config()
        self._apply_input_config()
        self._apply_mask_head_config()
        self._apply_test_config()
        
        # Output directory
        output_dir = self.config_data.get("output_dir", "output")
        self.cfg.OUTPUT_DIR = str(self.project_root / output_dir)
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        
        logger.info("Custom configurations applied successfully")
    
    # [Include other necessary methods - abbreviated for space]
    def _apply_dataset_config(self):
        datasets_config = self.config_data.get("datasets", {})
        self.cfg.DATASETS.TRAIN = ("custom_train",)
        self.cfg.DATASETS.TEST = ("custom_val",)
    
    def _apply_dataloader_config(self):
        dataloader_config = self.config_data.get("dataloader", {})
        if "num_workers" in dataloader_config:
            self.cfg.DATALOADER.NUM_WORKERS = dataloader_config["num_workers"]
    
    def _apply_solver_config(self):
        solver_config = self.config_data.get("solver", {})
        required_params = ["ims_per_batch", "base_lr", "max_iter"]
        for param in required_params:
            if param not in solver_config:
                raise ConfigurationError(f"Missing required solver parameter: {param}")
        
        self.cfg.SOLVER.IMS_PER_BATCH = solver_config["ims_per_batch"]
        self.cfg.SOLVER.BASE_LR = solver_config["base_lr"]
        self.cfg.SOLVER.MAX_ITER = solver_config["max_iter"]
        
        if "warmup_iters" in solver_config:
            self.cfg.SOLVER.WARMUP_ITERS = solver_config["warmup_iters"]
        if "warmup_factor" in solver_config:
            self.cfg.SOLVER.WARMUP_FACTOR = solver_config["warmup_factor"]
        if "steps" in solver_config:
            self.cfg.SOLVER.STEPS = tuple(solver_config["steps"])
        if "checkpoint_period" in solver_config:
            self.cfg.SOLVER.CHECKPOINT_PERIOD = solver_config["checkpoint_period"]
    
    def _apply_model_config(self):
        model_config = self.config_data.get("model", {})
        
        if "anchor_generator" in model_config:
            anchor_config = model_config["anchor_generator"]
            if "sizes" in anchor_config:
                self.cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_config["sizes"]
            if "aspect_ratios" in anchor_config:
                self.cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = anchor_config["aspect_ratios"]
        
        if "fpn" in model_config:
            fpn_config = model_config["fpn"]
            if "fuse_type" in fpn_config:
                self.cfg.MODEL.FPN.FUSE_TYPE = fpn_config["fuse_type"]
            if "norm" in fpn_config:
                self.cfg.MODEL.FPN.NORM = fpn_config["norm"]
        
        if "resnets" in model_config:
            resnets_config = model_config["resnets"]
            if "deform_on_per_stage" in resnets_config:
                self.cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE = resnets_config["deform_on_per_stage"]
    
    def _apply_roi_heads_config(self):
        roi_config = self.config_data.get("roi_heads", {})
        if "num_classes" not in roi_config:
            raise ConfigurationError("Missing required parameter: roi_heads.num_classes")
        
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = roi_config["num_classes"]
        
        if "batch_size_per_image" in roi_config:
            self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_config["batch_size_per_image"]
        if "positive_fraction" in roi_config:
            self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = roi_config["positive_fraction"]
        if "SCORE_THRESH_TEST" in roi_config:
            self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_config["SCORE_THRESH_TEST"]
        if "NMS_THRESH_TEST" in roi_config:
            self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = roi_config["NMS_THRESH_TEST"]
        if "DETECTIONS_PER_IMAGE" in roi_config:
            self.cfg.TEST.DETECTIONS_PER_IMAGE = roi_config["DETECTIONS_PER_IMAGE"]
    
    def _apply_input_config(self):
        input_config = self.config_data.get("INPUT", {})
        
        if "MIN_SIZE_TRAIN" in input_config:
            min_size_train = input_config["MIN_SIZE_TRAIN"]
            if isinstance(min_size_train, (list, tuple)):
                self.cfg.INPUT.MIN_SIZE_TRAIN = list(min_size_train)
            else:
                self.cfg.INPUT.MIN_SIZE_TRAIN = [min_size_train]
        
        if "MAX_SIZE_TRAIN" in input_config:
            self.cfg.INPUT.MAX_SIZE_TRAIN = input_config["MAX_SIZE_TRAIN"]
        
        if "MIN_SIZE_TEST" in input_config:
            min_size_test = input_config["MIN_SIZE_TEST"]
            if isinstance(min_size_test, (list, tuple)):
                self.cfg.INPUT.MIN_SIZE_TEST = list(min_size_test)[0]
            else:
                self.cfg.INPUT.MIN_SIZE_TEST = min_size_test
        
        if "MAX_SIZE_TEST" in input_config:
            self.cfg.INPUT.MAX_SIZE_TEST = input_config["MAX_SIZE_TEST"]
        
        # Critical: Set mask format for PointRend compatibility
        if self._should_use_pointrend():
            self.cfg.INPUT.MASK_FORMAT = "bitmask"
            logger.info("Set INPUT.MASK_FORMAT to 'bitmask' for PointRend compatibility")
        else:
            # Use polygon format for standard Mask R-CNN (more memory efficient)
            self.cfg.INPUT.MASK_FORMAT = "polygon"
    
    def _apply_mask_head_config(self):
        # Handle both old direct mask_head config and new nested roi_heads.mask_head config
        mask_config = self.config_data.get("mask_head", {})
        
        # Also check for nested mask head config in roi_heads (used by mask head hypertuning wrapper)
        roi_heads_config = self.config_data.get("roi_heads", {})
        if "mask_head" in roi_heads_config:
            nested_mask_config = roi_heads_config["mask_head"]
            # Convert from hypertuning wrapper format to config format
            if "num_conv" in nested_mask_config:
                mask_config["NUM_CONV"] = nested_mask_config["num_conv"]
            if "loss_weight" in nested_mask_config:
                mask_config["LOSS_WEIGHT"] = nested_mask_config["loss_weight"]
        
        if "NUM_CONV" in mask_config:
            self.cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = mask_config["NUM_CONV"]
            logger.info(f"Set mask head num_conv: {mask_config['NUM_CONV']}")
        if "POOLER_RESOLUTION" in mask_config:
            self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = mask_config["POOLER_RESOLUTION"]
        if "POOLER_SAMPLING_RATIO" in mask_config:
            self.cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = mask_config["POOLER_SAMPLING_RATIO"]
        if "LOSS_WEIGHT" in mask_config:
            self.cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = mask_config["LOSS_WEIGHT"]
            logger.info(f"Set mask head loss_weight: {mask_config['LOSS_WEIGHT']}")
    
    def _apply_test_config(self):
        test_config = self.config_data.get("TEST", {})
        
        if "AUG" in test_config:
            aug_config = test_config["AUG"]
            if aug_config.get("ENABLED", False):
                self.cfg.TEST.AUG.ENABLED = True
                if "MIN_SIZES" in aug_config:
                    self.cfg.TEST.AUG.MIN_SIZES = aug_config["MIN_SIZES"]
                if "MAX_SIZE" in aug_config:
                    self.cfg.TEST.AUG.MAX_SIZE = aug_config["MAX_SIZE"]
                if "FLIP" in aug_config:
                    self.cfg.TEST.AUG.FLIP = aug_config["FLIP"]
    
    def _apply_pointrend_configurations(self):
        """Apply PointRend-specific configurations with validation."""
        # Double-check PointRend is available before applying configs
        is_available, error_msg = self._check_pointrend_availability()
        if not is_available:
            logger.error(f"❌ Cannot apply PointRend configurations: {error_msg}")
            return
            
        pointrend_config = self.config_data.get("model", {}).get("pointrend", {})
        
        try:
            # Set PointRend-specific model components
            self.cfg.MODEL.ROI_HEADS.NAME = "PointRendROIHeads"
            self.cfg.MODEL.ROI_MASK_HEAD.NAME = "PointRendMaskHead"
            
            # Essential: Configure mask head to use PointRend
            self.cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE = ""  # No pooling for PointRend
            
            # Configure Point Head (this was missing and causing the error)
            self.cfg.MODEL.POINT_HEAD.NUM_CLASSES = self.cfg.MODEL.ROI_HEADS.NUM_CLASSES
            self.cfg.MODEL.POINT_HEAD.IN_FEATURES = pointrend_config.get("in_features", ["p2", "p3", "p4", "p5"])
            self.cfg.MODEL.POINT_HEAD.NUM_CONV = pointrend_config.get("num_conv", 3)
            self.cfg.MODEL.POINT_HEAD.CONV_DIM = pointrend_config.get("conv_dim", 256)
            self.cfg.MODEL.POINT_HEAD.NUM_POINTS = pointrend_config.get("num_points", 196)
            self.cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = pointrend_config.get("oversample_ratio", 3.0)
            self.cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = pointrend_config.get("importance_sample_ratio", 0.75)
            self.cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = pointrend_config.get("subdivision_steps", 5)
            self.cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = pointrend_config.get("subdivision_num_points", 784)
            
            # Critical: Set the mask point subdivision steps that was missing
            self.cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = True
            
            logger.info("✅ PointRend configurations applied successfully")
            logger.info(f"   Point Head Conv Layers: {self.cfg.MODEL.POINT_HEAD.NUM_CONV}")
            logger.info(f"   Subdivision Steps: {self.cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS}")
            logger.info(f"   Subdivision Points: {self.cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS}")
            
        except Exception as e:
            logger.error(f"❌ Failed to apply PointRend configurations: {str(e)}")
            logger.error("Falling back to standard Mask R-CNN configuration")
            
            # Reset to standard configuration
            self.cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
            self.cfg.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
            
            # Disable PointRend in the config
            pointrend_config["enabled"] = False
    
    def _validate_configuration(self):
        """Validate final configuration for consistency and completeness."""
        logger.info("Validating configuration...")
        
        if self.cfg.MODEL.ROI_HEADS.NUM_CLASSES != self.config_data["roi_heads"]["num_classes"]:
            raise ConfigurationError(
                f"NUM_CLASSES mismatch: expected {self.config_data['roi_heads']['num_classes']}, "
                f"got {self.cfg.MODEL.ROI_HEADS.NUM_CLASSES}"
            )
        
        if self.cfg.SOLVER.MAX_ITER <= 0:
            raise ConfigurationError(f"Invalid MAX_ITER: {self.cfg.SOLVER.MAX_ITER}")
        
        if self.cfg.SOLVER.BASE_LR <= 0:
            raise ConfigurationError(f"Invalid BASE_LR: {self.cfg.SOLVER.BASE_LR}")
        
        if self._should_use_pointrend():
            if self.cfg.MODEL.ROI_HEADS.NAME != "PointRendROIHeads":
                raise ConfigurationError("PointRend enabled but ROI_HEADS.NAME not set correctly")
            if self.cfg.MODEL.POINT_HEAD.NUM_CLASSES != self.cfg.MODEL.ROI_HEADS.NUM_CLASSES:
                raise ConfigurationError("PointRend NUM_CLASSES mismatch")
        
        logger.info("Configuration validation passed")


class DatasetManager:
    """Manages dataset registration and validation."""
    
    def __init__(self, config_data: Dict, project_root: Path):
        self.config_data = config_data
        self.project_root = project_root
        
    def register_datasets(self) -> Tuple[str, str]:
        """Register training and validation datasets."""
        logger.info("Registering datasets...")
        
        # Resolve dataset paths
        train_json = self._resolve_path(self.config_data["datasets"]["train_json"])
        val_json = self._resolve_path(self.config_data["datasets"]["val_json"])
        
        # Get image directories
        image_dirs = self._get_image_directories()
        
        # Validate paths exist
        self._validate_dataset_paths(train_json, val_json, image_dirs)
        
        # Dataset names
        train_name = "custom_train"
        val_name = "custom_val"
        
        # Register datasets if not already registered
        if train_name not in DatasetCatalog.list():
            train_data = self._load_coco_with_search(train_json, image_dirs, train_name)
            DatasetCatalog.register(train_name, lambda: train_data)
            MetadataCatalog.get(train_name).set(
                json_file=train_json, 
                image_root="", 
                evaluator_type="coco"
            )
            logger.info(f"Registered training dataset: {train_name} ({len(train_data)} samples)")
        
        if val_name not in DatasetCatalog.list():
            val_data = self._load_coco_with_search(val_json, image_dirs, val_name)
            DatasetCatalog.register(val_name, lambda: val_data)
            MetadataCatalog.get(val_name).set(
                json_file=val_json, 
                image_root="", 
                evaluator_type="coco"
            )
            logger.info(f"Registered validation dataset: {val_name} ({len(val_data)} samples)")
        
        return train_name, val_name
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve path relative to project root."""
        if os.path.isabs(path):
            return Path(path)
        return self.project_root / path
    
    def _get_image_directories(self) -> List[Path]:
        """Get list of possible image directories."""
        datasets_config = self.config_data["datasets"]
        
        if "image_dirs" in datasets_config:
            return [self._resolve_path(p) for p in datasets_config["image_dirs"]]
        else:
            dirs = []
            if "train_img_dir" in datasets_config:
                dirs.append(self._resolve_path(datasets_config["train_img_dir"]))
            if "val_img_dir" in datasets_config:
                dirs.append(self._resolve_path(datasets_config["val_img_dir"]))
            return dirs
    
    def _validate_dataset_paths(self, train_json: Path, val_json: Path, image_dirs: List[Path]):
        """Validate that required dataset files exist."""
        if not train_json.exists():
            raise ConfigurationError(f"Training annotation file not found: {train_json}")
        if not val_json.exists():
            raise ConfigurationError(f"Validation annotation file not found: {val_json}")
        
        for img_dir in image_dirs:
            if not img_dir.exists():
                logger.warning(f"Image directory does not exist: {img_dir}")
    
    def _load_coco_with_search(self, json_file: Path, image_dirs: List[Path], dataset_name: str) -> List[Dict]:
        """Load COCO annotations and find correct image paths."""
        logger.info(f"Loading COCO dataset from {json_file}")
        
        dataset_dicts = load_coco_json(str(json_file), image_root="", dataset_name=dataset_name)
        
        missing_files = []
        found_files = 0
        
        for d in dataset_dicts:
            file_name = d["file_name"]
            found_path = None
            
            for img_dir in image_dirs:
                path = img_dir / file_name
                if path.exists():
                    found_path = str(path)
                    break
            
            if found_path:
                d["file_name"] = found_path
                found_files += 1
            else:
                missing_files.append(file_name)
                d["file_name"] = str(image_dirs[0] / file_name)
        
        if missing_files:
            logger.warning(f"Missing {len(missing_files)} image files out of {len(dataset_dicts)} total")
            if len(missing_files) <= 10:
                logger.warning(f"Missing files: {missing_files}")
        
        logger.info(f"Successfully found {found_files}/{len(dataset_dicts)} image files")
        return dataset_dicts


def build_custom_augmentation() -> List[T.Augmentation]:
    """Build enhanced augmentation pipeline for segmentation."""
    return [
        T.RandomFlip(horizontal=True, vertical=False),
        T.RandomRotation(angle=[-10, 10]),
        T.RandomCrop("relative_range", (0.8, 0.8)),
        T.ResizeShortestEdge(
            short_edge_length=(640, 800, 960, 1024),
            max_size=1333,
            sample_style="choice",
        ),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
    ]


class CustomTrainer(DefaultTrainer):
    """Enhanced trainer with early stopping and better data loading."""
    
    def __init__(self, cfg, early_stopping_config: Optional[Dict] = None, val_dataset_name: str = "custom_val"):
        super().__init__(cfg)
        self.data_mapper = None
        
        # Initialize early stopping hook if configured
        if early_stopping_config and early_stopping_config.get("enabled", False):
            self.early_stopping_hook = EarlyStoppingHook(cfg, val_dataset_name, early_stopping_config)
            self.register_hooks([self.early_stopping_hook])
            logger.info("Early stopping hook registered successfully")
        else:
            self.early_stopping_hook = None
            logger.info("Early stopping disabled")
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with custom augmentation and error handling."""
        return build_detection_train_loader(cfg, mapper=RobustDatasetMapper(cfg, is_train=True))
    
    def run_step(self):
        """Override run_step to add better error monitoring and early stopping."""
        try:
            super().run_step()
        except StopTraining as e:
            logger.info(f"Training stopped by early stopping: {str(e)}")
            # Set a flag to stop the training loop gracefully
            self._should_stop = True
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory! Consider reducing batch size or image resolution.")
                raise
            else:
                logger.error(f"Training step failed: {str(e)}")
                raise
    
    def train(self):
        """Override train method to handle early stopping gracefully."""
        try:
            # Set up the stopping flag
            self._should_stop = False
            
            # Call parent train method but handle early stopping
            super().train()
            
        except StopTraining:
            logger.info("Training completed via early stopping")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Ensure early stopping hook gets proper cleanup
            if self.early_stopping_hook:
                logger.info("Training finished - early stopping hook cleanup")


class GPUMonitor:
    """Efficient GPU monitoring with proper logging and resource management."""
    
    def __init__(self, interval: int = 30, log_to_file: bool = True, output_dir: str = "output"):
        self.interval = max(interval, 10)
        self.log_to_file = log_to_file
        self.output_dir = Path(output_dir)
        self.stop_event = threading.Event()
        self.monitor_thread = None
        
        # Setup logging
        self.logger = logging.getLogger("gpu_monitor")
        if log_to_file:
            self.output_dir.mkdir(exist_ok=True)
            handler = logging.FileHandler(self.output_dir / "gpu_monitor.log")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Check if nvidia-smi is available
        self.nvidia_smi_available = self._check_nvidia_smi()
        self.last_memory_warning = 0
        self.memory_warning_cooldown = 300
        
    def _check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available and working."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], 
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _get_gpu_info(self) -> Optional[Dict]:
        """Get structured GPU information."""
        if not self.nvidia_smi_available:
            return None
            
        try:
            result = subprocess.run([
                "nvidia-smi", 
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return None
                
            gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info.append({
                            'index': int(parts[0]),
                            'name': parts[1],
                            'temperature': int(parts[2]) if parts[2] != 'N/A' else None,
                            'utilization': int(parts[3]) if parts[3] != 'N/A' else None,
                            'memory_used': int(parts[4]) if parts[4] != 'N/A' else None,
                            'memory_total': int(parts[5]) if parts[5] != 'N/A' else None,
                            'power_draw': float(parts[6]) if len(parts) > 6 and parts[6] != 'N/A' else None,
                            'power_limit': float(parts[7]) if len(parts) > 7 and parts[7] != 'N/A' else None,
                        })
            
            return {'gpus': gpu_info, 'timestamp': time.time()}
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            self.logger.warning(f"Failed to get GPU info: {str(e)}")
            return None
    
    def start(self):
        """Start GPU monitoring in background thread."""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU monitoring thread started")
    
    def stop(self):
        """Stop GPU monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.info("Stopping GPU monitoring...")
            self.stop_event.set()
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        logger.info(f"GPU monitoring started (interval: {self.interval}s)")
        
        if not self.nvidia_smi_available:
            logger.warning("nvidia-smi not available - GPU monitoring disabled")
            return
        
        iteration = 0
        while not self.stop_event.is_set():
            try:
                gpu_info = self._get_gpu_info()
                
                if gpu_info and iteration % 10 == 0:
                    status = self._format_gpu_status(gpu_info)
                    logger.info(f"GPU Status:\n{status}")
                    
                    if self.log_to_file:
                        self.logger.info(f"GPU Status:\n{status}")
                
                iteration += 1
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {str(e)}")
            
            self.stop_event.wait(self.interval)
        
        logger.info("GPU monitoring stopped")
    
    def _format_gpu_status(self, gpu_info: Dict) -> str:
        """Format GPU information for logging."""
        if not gpu_info or not gpu_info['gpus']:
            return "No GPU information available"
        
        status_lines = []
        for gpu in gpu_info['gpus']:
            memory_pct = (gpu['memory_used'] / gpu['memory_total'] * 100) if gpu['memory_used'] and gpu['memory_total'] else 0
            
            status = f"GPU {gpu['index']} ({gpu['name']}): "
            status += f"Util: {gpu['utilization'] or 'N/A'}%, "
            status += f"Mem: {gpu['memory_used'] or 'N/A'}MB/{gpu['memory_total'] or 'N/A'}MB ({memory_pct:.1f}%), "
            status += f"Temp: {gpu['temperature'] or 'N/A'}°C"
            
            if gpu['power_draw'] and gpu['power_limit']:
                power_pct = gpu['power_draw'] / gpu['power_limit'] * 100
                status += f", Power: {gpu['power_draw']:.1f}W/{gpu['power_limit']:.1f}W ({power_pct:.1f}%)"
            
            status_lines.append(status)
        
        return "\n".join(status_lines)
    
    def get_current_status(self) -> Optional[Dict]:
        """Get current GPU status synchronously."""
        return self._get_gpu_info()


def main():
    """Main training function with robust error handling and early stopping for mask head experiments."""
    parser = argparse.ArgumentParser(description="Robust Detectron2 Training System for Mask Head Hypertuning")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training if checkpoint exists")
    parser.add_argument("--validate_only", action="store_true", help="Only run validation")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping for this run")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Determine project root - use PWD to respect symlinks since Optuna runs from project root
        import os
        project_root = Path(os.environ.get('PWD', Path.cwd()))
        logger.info(f"Project root: {project_root}")
        
        # ============================================================================
        # EXPERIMENT IDENTIFICATION - CLEAR LOGGING FOR SLURM TRACKING
        # ============================================================================
        
        # Extract parameters from config for experiment identification
        learning_rate = config_data.get("solver", {}).get("base_lr", "unknown")
        batch_size = config_data.get("solver", {}).get("ims_per_batch", "unknown")
        output_dir = args.output_dir or config_data.get("output_dir", "output")
        
        # Extract mask head parameters from config
        mask_head_config = config_data.get("roi_heads", {}).get("mask_head", {})
        num_conv = mask_head_config.get("num_conv", config_data.get("mask_head", {}).get("NUM_CONV", "unknown"))
        loss_weight = mask_head_config.get("loss_weight", config_data.get("mask_head", {}).get("LOSS_WEIGHT", "unknown"))
        
        # Extract PointRend parameters from config
        pointrend_config = config_data.get("model", {}).get("pointrend", {})
        pointrend_enabled = pointrend_config.get("enabled", False)
        subdivision_steps = pointrend_config.get("subdivision_steps", "unknown")
        subdivision_points = pointrend_config.get("subdivision_num_points", "unknown")
        importance_ratio = pointrend_config.get("importance_sample_ratio", "unknown")
        
        # Extract ROI sampling parameters from config
        roi_heads_config = config_data.get("roi_heads", {})
        roi_batch_size_per_image = roi_heads_config.get("batch_size_per_image", "unknown")
        roi_positive_fraction = roi_heads_config.get("positive_fraction", "unknown")
        roi_score_thresh_test = roi_heads_config.get("SCORE_THRESH_TEST", "unknown")
        
        # Determine experiment type based on output directory pattern
        experiment_info = ""
        experiment_number = "N/A"
        experiment_type = "single"  # single, mask_head, pointrend, batch_size, or learning_rate
        primary_param = f"LR:{learning_rate}, Batch:{batch_size}, MaskHead:{num_conv}conv/{loss_weight}loss"
        param_name = "Training Config"
        
        # Check for PointRend grid search pattern
        if "pointrend_" in str(output_dir) and "_exp_" in str(output_dir):
            # Extract experiment info from PointRend output directory pattern
            try:
                # Pattern: "pointrend_finer_subdivision_exp_1" 
                import re
                exp_pattern = r"exp_(\d+)"
                strategy_pattern = r"pointrend_(\w+)_exp_"
                
                exp_match = re.search(exp_pattern, str(output_dir))
                strategy_match = re.search(strategy_pattern, str(output_dir))
                
                if exp_match:
                    experiment_number = int(exp_match.group(1)) + 1  # Convert to 1-indexed
                    
                    if strategy_match:
                        strategy_name = strategy_match.group(1)
                        experiment_info = f"POINTREND HYPERTUNING - EXPERIMENT {experiment_number}"
                        experiment_type = "pointrend"
                        primary_param = f"Strategy: {strategy_name}"
                        param_name = "PointRend Strategy"
                    else:
                        experiment_info = f"POINTREND EXPERIMENT {experiment_number}"
                        experiment_type = "pointrend"
                        primary_param = f"Steps:{subdivision_steps}, Points:{subdivision_points}, Ratio:{importance_ratio}"
                        param_name = "PointRend Config"
                
            except (ValueError, AttributeError):
                experiment_info = "POINTREND EXPERIMENT (could not parse details)"
                experiment_type = "pointrend"
                primary_param = f"Steps:{subdivision_steps}, Points:{subdivision_points}, Ratio:{importance_ratio}"
                param_name = "PointRend Config"
        
        # Check for mask head grid search pattern
        elif "mask_head_" in str(output_dir) and "_exp_" in str(output_dir):
            # Extract experiment info from mask head output directory pattern
            try:
                # Pattern: "mask_head_deeper_mask_head_exp_1" 
                import re
                exp_pattern = r"exp_(\d+)"
                strategy_pattern = r"mask_head_(\w+)_exp_"
                
                exp_match = re.search(exp_pattern, str(output_dir))
                strategy_match = re.search(strategy_pattern, str(output_dir))
                
                if exp_match:
                    experiment_number = int(exp_match.group(1)) + 1  # Convert to 1-indexed
                    
                    if strategy_match:
                        strategy_name = strategy_match.group(1)
                        experiment_info = f"MASK HEAD HYPERTUNING - EXPERIMENT {experiment_number}"
                        experiment_type = "mask_head"
                        primary_param = f"Strategy: {strategy_name}"
                        param_name = "Mask Head Strategy"
                    else:
                        experiment_info = f"MASK HEAD EXPERIMENT {experiment_number}"
                        experiment_type = "mask_head"
                        primary_param = f"NumConv:{num_conv}, LossWeight:{loss_weight}"
                        param_name = "Mask Head Config"
                
            except (ValueError, AttributeError):
                experiment_info = "MASK HEAD EXPERIMENT (could not parse details)"
                experiment_type = "mask_head"
                primary_param = f"NumConv:{num_conv}, LossWeight:{loss_weight}"
                param_name = "Mask Head Config"
        
        # Check for batch size grid search pattern
        elif "_batch_" in str(output_dir) and "_exp_" in str(output_dir):
            # Extract experiment info from batch size output directory pattern
            try:
                # Pattern: "batch_8_exp_2" 
                import re
                batch_pattern = r"batch_(\d+)"
                exp_pattern = r"exp_(\d+)"
                
                batch_match = re.search(batch_pattern, str(output_dir))
                exp_match = re.search(exp_pattern, str(output_dir))
                
                if batch_match and exp_match:
                    detected_batch = int(batch_match.group(1))
                    experiment_number = int(exp_match.group(1)) + 1  # Convert to 1-indexed
                    experiment_info = f"BATCH SIZE GRID SEARCH - EXPERIMENT {experiment_number}"
                    experiment_type = "batch_size"
                    primary_param = detected_batch
                    param_name = "Batch Size"
                    
                    # Update batch size from directory (authoritative source)
                    batch_size = detected_batch
                
            except (ValueError, AttributeError):
                experiment_info = "BATCH SIZE EXPERIMENT (could not parse details)"
                experiment_type = "batch_size"
                primary_param = batch_size
                param_name = "Batch Size"
        
        elif "lr_" in str(output_dir) and "exp_" in str(output_dir):
            # Extract experiment info from learning rate output directory pattern
            try:
                # Pattern: "lr_0.00025_exp_2"
                import re
                lr_pattern = r"lr_([\d\.]+)"
                exp_pattern = r"exp_(\d+)"
                
                lr_match = re.search(lr_pattern, str(output_dir))
                exp_match = re.search(exp_pattern, str(output_dir))
                
                if lr_match and exp_match:
                    detected_lr = float(lr_match.group(1))
                    experiment_number = int(exp_match.group(1)) + 1  # Convert to 1-indexed
                    experiment_info = f"LEARNING RATE GRID SEARCH - EXPERIMENT {experiment_number}"
                    experiment_type = "learning_rate"
                    primary_param = detected_lr
                    param_name = "Learning Rate"
                    
                    # Verify LR consistency
                    if abs(detected_lr - learning_rate) > 1e-8:
                        logger.warning(f"LR mismatch: config={learning_rate}, directory={detected_lr}")
                        learning_rate = detected_lr  # Use the one from directory as source of truth
                        primary_param = detected_lr
                
            except (ValueError, AttributeError):
                experiment_info = "LEARNING RATE EXPERIMENT (could not parse details)"
                experiment_type = "learning_rate"
                primary_param = learning_rate
                param_name = "Learning Rate"
        
        # Print prominent experiment header
        header_line = "=" * 80
        logger.info("")
        logger.info(header_line)
        logger.info("🚀 DETECTRON2 TRAINING EXPERIMENT STARTING")
        logger.info(header_line)
        
        if experiment_info:
            logger.info(f"📊 {experiment_info}")
            logger.info(f"🎯 EXPERIMENT NUMBER: {experiment_number}")
        else:
            logger.info("📊 SINGLE TRAINING MODE")
        
        logger.info(f"🔧 {param_name.upper()}: {primary_param}")
        
        # Display other parameters only when they're not the focus of the experiment
        if experiment_type == "batch_size":
            # For batch size experiments, show other params but not batch size
            logger.info(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            logger.info(f"📈 LEARNING RATE: {learning_rate}")
        elif experiment_type == "learning_rate":
            # For learning rate experiments, show other params but not learning rate
            logger.info(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            logger.info(f"📦 BATCH SIZE: {batch_size}")
        elif experiment_type == "mask_head":
            # For mask head experiments, show other params but not mask head details
            logger.info(f"📦 BATCH SIZE: {batch_size}")
            logger.info(f"📈 LEARNING RATE: {learning_rate}")
        elif experiment_type == "pointrend":
            # For PointRend experiments, show other params but not PointRend details
            logger.info(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            logger.info(f"📦 BATCH SIZE: {batch_size}")
            logger.info(f"📈 LEARNING RATE: {learning_rate}")
            logger.info(f"🎯 POINTREND - Steps: {subdivision_steps}, Points: {subdivision_points}, Ratio: {importance_ratio}")
        # For single mode, primary param already shows everything, so no redundant display
        
        # Always show ROI sampling parameters (important for all experiments)
        logger.info(f"🎯 ROI SAMPLING - BatchPerImg: {roi_batch_size_per_image}, PosFrac: {roi_positive_fraction}, ScoreThresh: {roi_score_thresh_test}")
        
        logger.info(f"📁 OUTPUT DIRECTORY: {output_dir}")
        logger.info(f"⚙️  CONFIG FILE: {args.config}")
        logger.info(f"🕐 START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(header_line)
        logger.info("")
        
        # Also log to stdout for immediate visibility in SLURM logs
        print(f"\n{header_line}")
        print(f"🚀 DETECTRON2 TRAINING EXPERIMENT STARTING")
        print(f"{header_line}")
        
        if experiment_info:
            print(f"📊 {experiment_info}")
            print(f"🎯 EXPERIMENT NUMBER: {experiment_number}")
        else:
            print(f"📊 SINGLE TRAINING MODE")
            
        print(f"🔧 {param_name.upper()}: {primary_param}")
        
        # Display other parameters only when they're not the focus of the experiment
        if experiment_type == "batch_size":
            # For batch size experiments, show other params but not batch size
            print(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            print(f"📈 LEARNING RATE: {learning_rate}")
        elif experiment_type == "learning_rate":
            # For learning rate experiments, show other params but not learning rate
            print(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            print(f"📦 BATCH SIZE: {batch_size}")
        elif experiment_type == "mask_head":
            # For mask head experiments, show other params but not mask head details
            print(f"📦 BATCH SIZE: {batch_size}")
            print(f"📈 LEARNING RATE: {learning_rate}")
        elif experiment_type == "pointrend":
            # For PointRend experiments, show other params but not PointRend details
            print(f"🎭 MASK HEAD - NumConv: {num_conv}, LossWeight: {loss_weight}")
            print(f"📦 BATCH SIZE: {batch_size}")
            print(f"📈 LEARNING RATE: {learning_rate}")
            print(f"🎯 POINTREND - Steps: {subdivision_steps}, Points: {subdivision_points}, Ratio: {importance_ratio}")
        # For single mode, primary param already shows everything, so no redundant display
        
        # Always show ROI sampling parameters (important for all experiments)
        print(f"🎯 ROI SAMPLING - BatchPerImg: {roi_batch_size_per_image}, PosFrac: {roi_positive_fraction}, ScoreThresh: {roi_score_thresh_test}")
        
        print(f"📁 OUTPUT DIRECTORY: {output_dir}")
        print(f"⚙️  CONFIG FILE: {args.config}")
        print(f"🕐 START TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{header_line}\n")
        
        # ============================================================================
        # CONTINUE WITH EXISTING LOGIC
        # ============================================================================
        
        # Override output directory if specified
        if args.output_dir:
            config_data["output_dir"] = args.output_dir
        
        # Override early stopping if disabled via command line
        if args.disable_early_stopping:
            if "early_stopping" in config_data:
                config_data["early_stopping"]["enabled"] = False
            logger.info("Early stopping disabled via command line argument")
        
        # Standard single training mode (removed grid search mode)
        logger.info("=== SINGLE TRAINING MODE ===")
        
        # Initialize managers
        config_manager = DetectronConfigManager(config_data, project_root)
        dataset_manager = DatasetManager(config_data, project_root)
        
        # Register datasets
        train_name, val_name = dataset_manager.register_datasets()
        
        # Build configuration
        cfg = config_manager.build_config()
        
        # Update dataset names in config
        cfg.DATASETS.TRAIN = (train_name,)
        cfg.DATASETS.TEST = (val_name,)
        
        # Get early stopping configuration
        early_stopping_config = config_data.get("early_stopping", {})
        
        logger.info("=== Configuration Summary ===")
        logger.info(f"Model: {cfg.MODEL.META_ARCHITECTURE}")
        logger.info(f"Backbone: {cfg.MODEL.BACKBONE.NAME}")
        logger.info(f"Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
        
        # Prominently display mask head configuration
        logger.info("=== Mask Head Configuration ===")
        logger.info(f"Mask Head Conv Layers: {cfg.MODEL.ROI_MASK_HEAD.NUM_CONV}")
        logger.info(f"Mask Head Loss Weight: {cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT}")
        logger.info(f"Mask Head Pooler Resolution: {cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION}")
        
        logger.info("=== Training Configuration ===")
        logger.info(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
        logger.info(f"Learning rate: {cfg.SOLVER.BASE_LR}")
        logger.info(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        logger.info(f"Output directory: {cfg.OUTPUT_DIR}")
        
        # Check if PointRend was requested vs. actually enabled
        pointrend_requested = config_data.get("model", {}).get("pointrend", {}).get("enabled", False)
        pointrend_actual = cfg.MODEL.ROI_HEADS.NAME == "PointRendROIHeads"
        
        if pointrend_actual:
            logger.info("=== PointRend Configuration ===")
            logger.info("✅ PointRend: ENABLED")
            logger.info(f"  Subdivision Steps: {cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS}")
            logger.info(f"  Subdivision Points: {cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS}")
            logger.info(f"  Importance Sample Ratio: {cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO}")
            logger.info(f"  Point Head Conv Layers: {cfg.MODEL.POINT_HEAD.NUM_CONV}")
            logger.info(f"  Point Head Conv Dim: {cfg.MODEL.POINT_HEAD.CONV_DIM}")
        elif pointrend_requested:
            logger.warning("=== PointRend Configuration ===")
            logger.warning("⚠️  PointRend: REQUESTED BUT NOT ENABLED")
            logger.warning("   PointRend was requested in config but fell back to standard Mask R-CNN")
            logger.warning("   This usually means PointRend is not properly installed")
            logger.warning("   To install PointRend:")
            logger.warning("     1. git clone https://github.com/facebookresearch/detectron2.git")
            logger.warning("     2. cd detectron2/projects/PointRend && pip install -e .")
            logger.warning("     3. Restart your training")
        else:
            logger.info("PointRend: DISABLED")
        
        # Early stopping summary
        if early_stopping_config.get("enabled", False):
            logger.info("=== Early Stopping Configuration ===")
            logger.info(f"Metric: {early_stopping_config.get('metric', 'segm/AP')}")
            logger.info(f"Patience: {early_stopping_config.get('patience', 4)}")
            logger.info(f"Min Delta: {early_stopping_config.get('min_delta', 0.001)}")
            logger.info(f"Eval Period: {early_stopping_config.get('eval_period', 500)} iterations")
            logger.info(f"Min Iterations: {early_stopping_config.get('min_iterations', 2000)}")
            logger.info(f"Restore Best Weights: {early_stopping_config.get('restore_best_weights', True)}")
        else:
            logger.info("Early Stopping: DISABLED")
        
        if args.validate_only:
            logger.info("Running validation only...")
            # Initialize trainer for validation
            trainer = CustomTrainer(cfg, early_stopping_config, val_name)
            checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
            
            evaluator = COCOEvaluator(val_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(cfg, val_name)
            metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
            
            logger.info("=== Validation Metrics ===")
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")
            return
        
        # Initialize GPU monitoring if enabled
        gpu_monitor = None
        if config_data.get("logging", {}).get("gpu_monitoring", False):
            interval = config_data["logging"].get("monitor_interval_sec", 30)
            gpu_monitor = GPUMonitor(
                interval=interval,
                log_to_file=True,
                output_dir=cfg.OUTPUT_DIR
            )
            gpu_monitor.start()
            
            # Log initial GPU status
            initial_status = gpu_monitor.get_current_status()
            if initial_status:
                logger.info("=== Initial GPU Status ===")
                logger.info(gpu_monitor._format_gpu_status(initial_status))
        
        # Initialize trainer with early stopping
        logger.info("Initializing trainer with early stopping support...")
        trainer = CustomTrainer(cfg, early_stopping_config, val_name)
        
        # Setup checkpointing
        logger.info("Setting up model checkpointing...")
        checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
        
        if args.resume:
            logger.info("Attempting to resume from checkpoint...")
            loaded = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
            if loaded:
                logger.info("Successfully resumed from existing checkpoint")
            else:
                logger.warning("No checkpoint found for resuming - starting fresh training")
        else:
            logger.info("Loading pretrained weights...")
            try:
                checkpointer.load(cfg.MODEL.WEIGHTS, checkpointables=[])
                logger.info("Pretrained weights loaded successfully")
                logger.info("Note: Shape mismatches are expected when fine-tuning with different classes")
                logger.info("Note: 'Parameters not found' warnings can occur if your config enables features not present in the pretrained model")
            except Exception as e:
                logger.error(f"Error loading pretrained weights: {str(e)}")
                logger.info("Proceeding with random initialization")
        
        # Start training
        training_start_time = time.time()
        
        logger.info("=== Starting Training with Early Stopping ===")
        logger.info("Note: Data augmentation pipeline includes:")
        logger.info("  - Random horizontal flip")
        logger.info("  - Random rotation (±10°)")
        logger.info("  - Random crop (0.8 relative range)")
        logger.info("  - Multi-scale resize (640-1024px)")
        logger.info("  - Color jittering")
        
        if early_stopping_config.get("enabled", False):
            logger.info("Early stopping will monitor validation performance and prevent overfitting")
            logger.info(f"Evaluation will run every {early_stopping_config.get('eval_period', 500)} iterations")
        
        # Print training start message to stdout for SLURM visibility
        if experiment_info:
            print(f"🚀 Starting Experiment {experiment_number} training with {param_name} {primary_param}...")
        else:
            print(f"🚀 Starting training with {param_name} {primary_param}...")
        
        try:
            trainer.train()
            training_time = time.time() - training_start_time
            
            logger.info("Training completed successfully!")
            print(f"✅ Training completed successfully in {training_time/60:.1f} minutes!")
            
            # Check if training was stopped early
            if hasattr(trainer, 'early_stopping_hook') and trainer.early_stopping_hook:
                if trainer.early_stopping_hook.should_stop:
                    logger.info("=== Early Stopping Summary ===")
                    logger.info(f"Training stopped early at iteration {trainer.iter}")
                    logger.info(f"Best {trainer.early_stopping_hook.metric_name}: {trainer.early_stopping_hook.best_metric:.6f}")
                    logger.info(f"Best iteration: {trainer.early_stopping_hook.best_iteration}")
                    logger.info("This prevented overfitting and saved computational resources!")
                else:
                    logger.info("Training completed full duration - early stopping was not triggered")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except StopTraining as e:
            logger.info(f"Training stopped early: {str(e)}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            # Stop GPU monitoring
            if gpu_monitor:
                gpu_monitor.stop()
        
        # Run final evaluation
        logger.info("=== Running Final Evaluation ===")
        try:
            evaluator = COCOEvaluator(val_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
            val_loader = build_detection_test_loader(cfg, val_name)
            metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
            
            logger.info("=== Final Evaluation Metrics ===")
            for key, value in metrics.items():
                logger.info(f"{key}: {value}")
            
            # Compare with best early stopping metrics if available
            if (hasattr(trainer, 'early_stopping_hook') and 
                trainer.early_stopping_hook and 
                trainer.early_stopping_hook.should_stop):
                
                hook = trainer.early_stopping_hook
                final_metric = metrics.get(hook.metric_name.replace('/', '_'), {})
                if isinstance(final_metric, dict):
                    final_value = final_metric.get(hook.metric_name.split('/')[-1], 0)
                else:
                    final_value = final_metric
                
                logger.info(f"Final {hook.metric_name}: {final_value:.6f}")
                logger.info(f"Best {hook.metric_name} (early stopping): {hook.best_metric:.6f}")
                
                if hook.restore_best_weights:
                    logger.info("Note: Model was restored to best weights from early stopping")
                
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
        
        # Final GPU status
        if gpu_monitor:
            final_status = gpu_monitor.get_current_status()
            if final_status:
                logger.info("=== Final GPU Status ===")
                logger.info(gpu_monitor._format_gpu_status(final_status))
        
        # Check for early stopping artifacts
        early_stopping_files = [
            "early_stopping_history.json",
            "early_stopping_summary.txt",
            "model_best.pth"
        ]
        
        available_files = []
        for filename in early_stopping_files:
            filepath = Path(cfg.OUTPUT_DIR) / filename
            if filepath.exists():
                available_files.append(filename)
        
        if available_files:
            logger.info("=== Early Stopping Artifacts Created ===")
            for filename in available_files:
                logger.info(f"  - {filename}")
        
        # ============================================================================
        # EXPERIMENT COMPLETION SUMMARY
        # ============================================================================
        
        total_experiment_time = time.time() - training_start_time if 'training_start_time' in locals() else 0
        
        completion_header = "=" * 80
        logger.info("")
        logger.info(completion_header)
        logger.info("🎉 DETECTRON2 TRAINING EXPERIMENT COMPLETED")
        logger.info(completion_header)
        
        if experiment_info:
            logger.info(f"📊 {experiment_info}")
            logger.info(f"🎯 EXPERIMENT NUMBER: {experiment_number}")
        else:
            logger.info("📊 SINGLE TRAINING MODE")
        
        logger.info(f"🔧 {param_name.upper()} TESTED: {primary_param}")
        logger.info(f"📁 RESULTS SAVED TO: {output_dir}")
        logger.info(f"⏱️  TOTAL EXPERIMENT TIME: {total_experiment_time/60:.1f} minutes")
        logger.info(f"🕐 COMPLETION TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(completion_header)
        logger.info("")
        
        # Also print completion summary to stdout for SLURM logs
        print(f"\n{completion_header}")
        print(f"🎉 DETECTRON2 TRAINING EXPERIMENT COMPLETED")
        print(f"{completion_header}")
        
        if experiment_info:
            print(f"📊 {experiment_info}")
            print(f"🎯 EXPERIMENT NUMBER: {experiment_number}")
        else:
            print(f"📊 SINGLE TRAINING MODE")
            
        print(f"🔧 {param_name.upper()} TESTED: {primary_param}")
        print(f"📁 RESULTS SAVED TO: {output_dir}")
        print(f"⏱️  TOTAL EXPERIMENT TIME: {total_experiment_time/60:.1f} minutes")
        print(f"🕐 COMPLETION TIME: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{completion_header}\n")
        
        logger.info("Training pipeline completed!")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {str(e)}")
        if 'experiment_info' in locals() and experiment_info:
            print(f"❌ Experiment {experiment_number} ({param_name} {primary_param}) failed: Configuration error")
        return 1
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        if 'experiment_info' in locals() and experiment_info:
            print(f"❌ Experiment {experiment_number} ({param_name} {primary_param}) failed: File not found")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        if 'experiment_info' in locals() and experiment_info:
            print(f"❌ Experiment {experiment_number} ({param_name} {primary_param}) failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
