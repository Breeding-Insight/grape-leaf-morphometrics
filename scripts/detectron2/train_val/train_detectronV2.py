import os
import argparse
import yaml
import torch
from pathlib import Path
from detectron2.engine import DefaultTrainer, launch
from detectron2.config import get_cfg, CfgNode as CN
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
import logging

class MaskRCNNTrainer(DefaultTrainer):
    """
    Enhanced trainer with proper evaluation and checkpoint handling.
    Like Hari Seldon's equations, we must ensure every component functions harmoniously.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build COCO evaluator for proper mAP computation."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        """Enhanced hooks for better training monitoring."""
        hooks = super().build_hooks()
        
        # Add evaluation hook with proper frequency
        from detectron2.evaluation import inference_on_dataset
        from detectron2.engine import HookBase
        
        class EvalHook(HookBase):
            def __init__(self, eval_period, model, data_loader, evaluator):
                self._period = eval_period
                self._model = model
                self._data_loader = data_loader
                self._evaluator = evaluator
            
            def _do_eval(self):
                results = inference_on_dataset(self._model, self._data_loader, self._evaluator)
                # Log results to console
                print(f"Evaluation results at iteration {self.trainer.iter}:")
                for task, metrics in results.items():
                    for metric, value in metrics.items():
                        print(f"  {task}/{metric}: {value:.4f}")
                return results
            
            def after_step(self):
                if (self.trainer.iter + 1) % self._period == 0:
                    self._do_eval()
        
        # Add evaluation every 500 iterations
        if len(cfg.DATASETS.TEST) > 0:
            eval_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
            evaluator = self.build_evaluator(cfg, cfg.DATASETS.TEST[0])
            hooks.insert(-1, EvalHook(500, self.model, eval_loader, evaluator))
        
        return hooks

def setup_detectron_config(config_data):
    """
    Configure Detectron2 with mathematical precision.
    Every parameter must be chosen with the care of a psychohistorian.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_data["model_zoo_config"]))
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("custom_train",)
    cfg.DATASETS.TEST = ("custom_val",)
    cfg.DATALOADER.NUM_WORKERS = config_data["dataloader"]["num_workers"]
    
    # Solver configuration with corrected learning rates
    cfg.SOLVER.IMS_PER_BATCH = config_data["solver"]["ims_per_batch"]
    cfg.SOLVER.BASE_LR = config_data["solver"]["base_lr"]
    cfg.SOLVER.MAX_ITER = config_data["solver"]["max_iter"]
    cfg.SOLVER.STEPS = tuple(config_data["solver"]["steps"])
    
    # Critical: Proper learning rate scheduling
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    cfg.SOLVER.WARMUP_ITERS = 500  # Warmup iterations
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000  # Warmup factor
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.MOMENTUM = 0.9
    
    # Model configuration
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config_data["roi_heads"]["batch_size_per_image"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config_data["roi_heads"]["num_classes"]
    
    # Critical thresholds for prediction generation
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for detection
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 100  # Allow more detections
    
    # RPN configuration for better proposal generation
    # Default values if not specified in config
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.7
    
    # Apply RPN configurations if provided
    if "rpn" in config_data:
        rpn_config = config_data["rpn"]
        if "PRE_NMS_TOPK_TRAIN" in rpn_config:
            cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = rpn_config["PRE_NMS_TOPK_TRAIN"]
        if "PRE_NMS_TOPK_TEST" in rpn_config:
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = rpn_config["PRE_NMS_TOPK_TEST"]
        if "POST_NMS_TOPK_TRAIN" in rpn_config:
            cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = rpn_config["POST_NMS_TOPK_TRAIN"]
        if "POST_NMS_TOPK_TEST" in rpn_config:
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = rpn_config["POST_NMS_TOPK_TEST"]
        if "NMS_THRESH" in rpn_config:
            cfg.MODEL.RPN.NMS_THRESH = rpn_config["NMS_THRESH"]
        if "POSITIVE_FRACTION" in rpn_config:
            cfg.MODEL.RPN.POSITIVE_FRACTION = rpn_config["POSITIVE_FRACTION"]
        if "BATCH_SIZE_PER_IMAGE" in rpn_config:
            cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = rpn_config["BATCH_SIZE_PER_IMAGE"]
    
    # Apply custom configurations if provided
    if "model" in config_data and "anchor_generator" in config_data["model"]:
        if "sizes" in config_data["model"]["anchor_generator"]:
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = config_data["model"]["anchor_generator"]["sizes"]
        if "aspect_ratios" in config_data["model"]["anchor_generator"]:
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = config_data["model"]["anchor_generator"]["aspect_ratios"]
    
    # Input configuration
    if "INPUT" in config_data:
        if "MIN_SIZE_TRAIN" in config_data["INPUT"]:
            min_sizes = config_data["INPUT"]["MIN_SIZE_TRAIN"]
            if isinstance(min_sizes, str):
                min_sizes = [int(x.strip()) for x in min_sizes.replace('(', '').replace(')', '').split(',')]
            cfg.INPUT.MIN_SIZE_TRAIN = min_sizes
        
        if "MAX_SIZE_TRAIN" in config_data["INPUT"]:
            cfg.INPUT.MAX_SIZE_TRAIN = config_data["INPUT"]["MAX_SIZE_TRAIN"]
        
        if "MIN_SIZE_TEST" in config_data["INPUT"]:
            min_sizes = config_data["INPUT"]["MIN_SIZE_TEST"]
            if isinstance(min_sizes, str):
                min_sizes = [int(x.strip()) for x in min_sizes.replace('(', '').replace(')', '').split(',')]
            cfg.INPUT.MIN_SIZE_TEST = min_sizes
            
        if "MAX_SIZE_TEST" in config_data["INPUT"]:
            cfg.INPUT.MAX_SIZE_TEST = config_data["INPUT"]["MAX_SIZE_TEST"]
    
    # Mask head configuration
    if "mask_head" in config_data:
        mask_config = config_data["mask_head"]
        if "NUM_CONV" in mask_config:
            cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = mask_config["NUM_CONV"]
        if "POOLER_RESOLUTION" in mask_config:
            cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = mask_config["POOLER_RESOLUTION"]
    
    # Loss weights for balanced training
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0
    cfg.MODEL.ROI_HEADS.LOSS_WEIGHT = 1.0
    
    # Set model weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_data["model_zoo_config"])
    
    # Output directory
    cfg.OUTPUT_DIR = config_data.get("output_dir", "./output")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Enable evaluation during training - respect config if provided
    if "logging" in config_data and "eval_period" in config_data["logging"]:
        cfg.TEST.EVAL_PERIOD = config_data["logging"]["eval_period"]
    else:
        cfg.TEST.EVAL_PERIOD = 500  # Default evaluation period
    
    return cfg

def validate_dataset_registration(train_json, train_img_dir, val_json, val_img_dir):
    """
    Validate dataset paths and registration.
    Like verifying the mathematical foundations before building the psychohistory equations.
    """
    # Check if files exist
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training annotation file not found: {train_json}")
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation annotation file not found: {val_json}")
    if not os.path.exists(train_img_dir):
        raise FileNotFoundError(f"Training image directory not found: {train_img_dir}")
    if not os.path.exists(val_img_dir):
        raise FileNotFoundError(f"Validation image directory not found: {val_img_dir}")
    
    print(f"✓ Training annotations: {train_json}")
    print(f"✓ Training images: {train_img_dir}")
    print(f"✓ Validation annotations: {val_json}")
    print(f"✓ Validation images: {val_img_dir}")

def main():
    """
    Main training function with enhanced error handling and monitoring.
    """
    # Setup logging
    setup_logger()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Enhanced Mask R-CNN training with proper evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training if checkpoint exists")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Resolve paths
    project_root = os.getcwd()
    train_json = os.path.join(project_root, config_data["datasets"]["train_json"])
    train_img_dir = os.path.join(project_root, config_data["datasets"]["train_img_dir"])
    val_json = os.path.join(project_root, config_data["datasets"]["val_json"])
    val_img_dir = os.path.join(project_root, config_data["datasets"]["val_img_dir"])
    
    # Validate dataset paths
    validate_dataset_registration(train_json, train_img_dir, val_json, val_img_dir)
    
    # Register datasets
    TRAIN_NAME = "custom_train"
    VAL_NAME = "custom_val"
    
    # Clear existing registrations
    if TRAIN_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(TRAIN_NAME)
        MetadataCatalog.remove(TRAIN_NAME)
    if VAL_NAME in DatasetCatalog.list():
        DatasetCatalog.remove(VAL_NAME)
        MetadataCatalog.remove(VAL_NAME)
    
    # Register datasets
    register_coco_instances(TRAIN_NAME, {}, train_json, train_img_dir)
    register_coco_instances(VAL_NAME, {}, val_json, val_img_dir)
    
    # Get dataset metadata
    train_metadata = MetadataCatalog.get(TRAIN_NAME)
    val_metadata = MetadataCatalog.get(VAL_NAME)
    
    print(f"Training dataset: {len(DatasetCatalog.get(TRAIN_NAME))} images")
    print(f"Validation dataset: {len(DatasetCatalog.get(VAL_NAME))} images")
    print(f"Number of classes: {len(train_metadata.thing_classes)}")
    print(f"Class names: {train_metadata.thing_classes}")
    
    # Setup configuration
    cfg = setup_detectron_config(config_data)
    
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Create trainer
    trainer = MaskRCNNTrainer(cfg)
    
    # Resume or load model
    trainer.resume_or_load(resume=args.resume)
    
    if args.eval_only:
        # Evaluation only
        evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(cfg, VAL_NAME)
        results = inference_on_dataset(trainer.model, val_loader, evaluator)
        print("=== Evaluation Results ===")
        for task, metrics in results.items():
            print(f"{task}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        return
    
    # Training
    print("=== Starting Training ===")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"LR steps: {cfg.SOLVER.STEPS}")
    
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Final evaluation
    print("=== Final Evaluation ===")
    evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, VAL_NAME)
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    
    print("=== Final Results ===")
    for task, metrics in results.items():
        print(f"{task}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()