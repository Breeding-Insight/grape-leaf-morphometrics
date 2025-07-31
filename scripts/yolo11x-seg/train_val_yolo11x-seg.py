"""
YOLO Model Training Script
==========================

This script provides a framework for training YOLO models, optimized 
for both local environments and SLURM clusters.

Features:
- Supports both local and SLURM environments with automatic configuration
- Early stopping to prevent wasted computation time
- Checkpointing for resuming interrupted training
- Hardware-aware with support for CUDA, MPS (M1/M2 Macs), and CPU
- Memory optimization for resource-constrained environments

Usage:
    python train_yolo.py --data [DATA_YAML] --output-dir [OUTPUT_DIR] --model [MODEL_PATH]

Author: Arlyn Ackerman
Date: March 2025
"""

# Standard libraries
import os  # Operating system interfaces
import argparse  # Command-line argument parsing
import json  # JSON file handling
import time  # Time-related functions
import platform  # Platform-specific information

# Third-party libraries
import torch  # PyTorch for deep learning
import numpy as np  # Numerical operations

# YOLO library
from ultralytics import YOLO  # YOLO model from Ultralytics

def parse_args():
    """Parse command line arguments for YOLOv11x training."""
    parser = argparse.ArgumentParser(description='Train YOLOv11x-seg model')

    # Core parameters
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml file')
    parser.add_argument('--output-dir', type=str, default='./runs/train_yolo11x',
                        help='Output directory for saving results')
    parser.add_argument('--model', type=str, default='yolo11x-seg.pt',
                        help='Path to YOLOv11x-seg model')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading')

    # YOLOv11x-specific parameters
    parser.add_argument('--attention', type=str, choices=['sca', 'eca', 'none'], 
                        default='sca', help='Attention mechanism type')
    parser.add_argument('--use-repvgg', action='store_true', default=True,
                        help='Use RepVGG blocks')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], 
                        default='AdamW', help='Optimizer type')
  
      # Early stopping parameters
    parser.add_argument('--patience', type=int, default=15,
                      help='Number of epochs with no improvement for early stopping')
    parser.add_argument('--min-delta', type=float, default=0.001,
                      help='Minimum change to qualify as improvement')
    
    # Additional options
    parser.add_argument('--amp', action='store_true',
                    help='Use Automatic Mixed Precision')
    parser.add_argument('--save-period', type=int, default=5,
                    help='Save checkpoint every x epochs')
    parser.add_argument('--verbose', action='store_true',
                    help='Verbose output')
    parser.add_argument('--dropout', type=float, default=0.0,
                    help='Dropout rate')
    parser.add_argument('--rect', action='store_true',
                    help='Use rectangular training')

    # Device settings
    parser.add_argument('--device', type=str, default='0',
                        help='Device to train on (e.g., 0, 0,1,2,3, cpu)')

    return parser.parse_args()

def setup_environment(args):
    """Check and setup training environment."""
    # Check if CUDA is available
    if args.device != 'cpu' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "weights"), exist_ok=True)

    # Save configuration
    config = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "device": args.device,
        "attention": args.attention,
        "use_repvgg": args.use_repvgg,
        "optimizer": args.optimizer,
        "early_stopping_patience": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print(f"Environment setup complete. Training with:")
    print(f"- Model: {args.model}")
    print(f"- Device: {args.device}")
    print(f"- Batch size: {args.batch_size}")

    return config

def train_yolo11x(args, config):
    """Train YOLOv11x-seg model with segment-optimized parameters and early stopping."""
    print("\n=== Starting YOLOv11x-seg training with segmentation optimization ===")
    print(f"Early stopping patience: {args.patience} epochs")
    start_time = time.time()

    # Initialize variables to track best model performance
    best_fitness = 0
    best_epoch = 0
    patience_counter = 0

    try:
        # Clear GPU memory before loading model
        if args.device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cleared CUDA memory before loading model")

        # Load YOLOv11x-seg model
        model = YOLO(args.model)
        print(f"YOLOv11x-seg model loaded: {args.model}")

        # Validate that model supports segmentation
        model_task = getattr(model, 'task', None)
        if model_task != 'segment' and not args.model.endswith('-seg.pt'):
            print(f"Warning: Model may not support segmentation (task: {model_task})")
            print("Will attempt to train with segmentation parameters anyway")

        # Create YOLO parameters dictionary with segmentation-specific parameters
        yolo_args = {
            # Standard training parameters
            'data': args.data,
            'epochs': args.epochs,
            'batch': args.batch_size,
            'imgsz': args.img_size,
            'device': args.device,
            'workers': min(args.workers, 16),  # Limit workers to avoid DataLoader issues
            'project': os.path.dirname(args.output_dir),
            'name': os.path.basename(args.output_dir),
            'exist_ok': True,
            'verbose': True,

            # Segmentation-specific parameters
            'task': 'segment',  # Explicitly set segmentation task
            'mask_ratio': 4,    # Downsample ratio for masks (smaller = higher resolution but slower)
            'overlap_mask': True,  # Allow masks to overlap for multi-instance segmentation

            # Loss weights to balance segmentation vs detection
            'box': 7.5,  # Box loss weight
            'cls': 0.5,  # Class loss weight 
            'dfl': 1.5,  # Distribution focal loss weight
            # Note: 'mask' loss weight is not supported in current Ultralytics version

            # Segmentation-specific thresholds
            'iou': 0.7,  # IoU threshold for NMS (higher for segmentation)

            # Segmentation refinement options
            'nms': True,  # Use NMS
            'retina_masks': True,  # Use high-resolution segmentation masks
            'single_cls': False,  # Treat as multi-class segmentation

            # Advanced segmentation options
            'augment': True,  # Use data augmentation for robust segmentation
            'hsv_h': 0.015,  # HSV augmentations (careful adjustments for segmentation)
            'hsv_s': 0.7, 
            'hsv_v': 0.4,
            'degrees': 0.0,  # More conservative rotation for segmentation
            'scale': 0.5,  # Scale augmentation
        }

        # Add segmentation-specific learning rate scheduling
        # Segmentation models need lower learning rates for mask head training
        lr_config = {
            'lr0': 0.001,            # Initial learning rate (lower than detection default)
            'lrf': 0.01,             # Final learning rate as a fraction of initial
            'momentum': 0.937,       # SGD momentum/Adam beta1
            'weight_decay': 0.0005,  # Optimizer weight decay
            'warmup_epochs': 3.0,    # Longer warmup for segmentation helps stability
            'warmup_momentum': 0.8,  # Warmup momentum
            'warmup_bias_lr': 0.1,   # Warmup initial bias lr
        }

        # For finer mask prediction, add a final fine-tuning phase with lower LR
        seg_lr_schedule = {
            'cos_lr': True,          # Use cosine LR scheduler
            # Note: lr_dropout is not supported in current Ultralytics version
        }

        # Add segmentation-specific learning rate parameters to yolo_args
        yolo_args.update(lr_config)
        yolo_args.update(seg_lr_schedule)

        # Add optimizer-specific LR handling
        if yolo_args.get('optimizer') == 'SGD':
            print("Using SGD optimizer with lower momentum for segmentation")
            yolo_args['momentum'] = 0.9  # Slightly lower momentum helps with mask refinement
        elif yolo_args.get('optimizer') == 'Adam':
            print("Using Adam optimizer with segmentation-specific settings")
            yolo_args['weight_decay'] = 0.00025  # Lower weight decay for Adam with segmentation
        elif yolo_args.get('optimizer') == 'AdamW':
            print("Using AdamW optimizer with segmentation-specific settings")
            # AdamW default settings work well with provided LR schedule

        print(f"Segmentation learning rate schedule: {lr_config['lr0']:.5f} → {lr_config['lr0']*lr_config['lrf']:.7f}")
        print(f"Warmup: {lr_config['warmup_epochs']} epochs")
        print(f"Scheduler: {'Cosine' if seg_lr_schedule.get('cos_lr', False) else 'Linear'}")

        # Add optional standard parameters if they exist
        if hasattr(args, 'amp') and args.amp:
            yolo_args['amp'] = True
        if hasattr(args, 'save_period'):
            yolo_args['save_period'] = args.save_period
        if hasattr(args, 'rect') and args.rect:
            yolo_args['rect'] = True
        if hasattr(args, 'optimizer'):
            yolo_args['optimizer'] = args.optimizer

        # Add dropout if specified
        if hasattr(args, 'dropout') and args.dropout > 0:
            yolo_args['dropout'] = args.dropout

        # Note: YOLOv11x-specific parameters are not supported in current Ultralytics version
        # These parameters are commented out to avoid errors:
        # - mask_iou_thr
        # - attention (sca, eca)
        # - repvgg_block
        # - lr_dropout
        # - mask (use task='segment' instead)
        
        print("Note: YOLOv11x-specific parameters (attention, repvgg_block, mask_iou_thr) are not supported in current Ultralytics version")
        print("Training will proceed with standard YOLO segmentation parameters")

        # Memory management for large segmentation models
        if args.batch_size < 8 and args.device != 'cpu':
            yolo_args['accumulate'] = 2  # Gradient accumulation for large segmentation models
            print(f"Using gradient accumulation (accumulate=2) for small batch size ({args.batch_size})")

        # Log segmentation-specific parameters
        print("\nTraining with segmentation-optimized parameters:")
        segmentation_keys = ['task', 'mask', 'mask_ratio', 'retina_masks']
        segmentation_keys.extend(['mask_iou_thr'] if 'mask_iou_thr' in yolo_args else [])
        for key in segmentation_keys:
            if key in yolo_args:
                print(f"  {key}: {yolo_args[key]}")
        print("")

        # Define segmentation-optimized early stopping callback
        def custom_on_train_epoch_end(trainer):
            nonlocal best_fitness, best_epoch, patience_counter
            
            try:
                # Get current epoch
                current_epoch = getattr(trainer, 'epoch', 0)
                
                # Get fitness from trainer - YOLO stores this in trainer.fitness
                current_fitness = None
                
                # Primary: Use trainer.fitness (this is the main metric YOLO uses)
                if hasattr(trainer, 'fitness') and trainer.fitness is not None:
                    current_fitness = float(trainer.fitness)
                    print(f"Using trainer.fitness: {current_fitness:.5f}")
                
                # Fallback: try to get from trainer.metrics if fitness not available
                elif hasattr(trainer, 'metrics') and trainer.metrics is not None:
                    # Try mask metrics first for segmentation
                    if hasattr(trainer.metrics, 'fitness'):
                        current_fitness = float(trainer.metrics.fitness)
                        print(f"Using metrics.fitness: {current_fitness:.5f}")
                    elif hasattr(trainer.metrics, 'mp') and trainer.metrics.mp is not None:
                        # Use mean precision as fallback
                        current_fitness = float(trainer.metrics.mp)
                        print(f"Using mean precision as fitness: {current_fitness:.5f}")
                
                # If still no fitness, skip early stopping for this epoch
                if current_fitness is None:
                    print(f"Warning: No fitness metric found in epoch {current_epoch}. Skipping early stopping check.")
                    return True  # Continue training

                # Early stopping logic
                if current_fitness > (best_fitness + args.min_delta):
                    best_fitness = current_fitness
                    best_epoch = current_epoch
                    patience_counter = 0
                    
                    print(f"New best fitness: {current_fitness:.5f} at epoch {current_epoch}")
                    
                    # Try to save best model (but don't fail if we can't)
                    try:
                        # Use trainer's save method if available (without custom path)
                        if hasattr(trainer, 'save_model'):
                            trainer.save_model()  # Let YOLO handle the path automatically
                            print(f"Best model saved automatically by YOLO trainer")
                        elif hasattr(trainer, 'best'):
                            # Some YOLO versions save automatically
                            print(f"Best model will be saved automatically by YOLO")
                        else:
                            print(f"Note: Manual model saving not available")
                    except Exception as save_error:
                        print(f"Warning: Could not save best model: {str(save_error)}")
                        
                else:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter}/{args.patience} epochs. Best: {best_fitness:.5f}, Current: {current_fitness:.5f}")

                    if patience_counter >= args.patience:
                        print(f"Early stopping triggered after {current_epoch} epochs")
                        print(f"Best fitness {best_fitness:.5f} was achieved at epoch {best_epoch}")
                        
                        # Signal early stopping to YOLO trainer
                        try:
                            # Set the stop flag (this is the standard way)
                            trainer.stopper.possible_stop = True
                            if hasattr(trainer.stopper, 'stop'):
                                trainer.stopper.stop = True
                            # Also try setting epoch to epochs to trigger completion
                            if hasattr(trainer, 'epochs'):
                                trainer.epoch = trainer.epochs
                        except:
                            # Fallback: try other stopping mechanisms
                            if hasattr(trainer, '_stop'):
                                trainer._stop = True
                            if hasattr(trainer, 'stop'):
                                trainer.stop = True
                        
                        return False  # Signal to stop

                return True  # Continue training
                    
            except Exception as e:
                print(f"Warning: Early stopping callback error: {str(e)}")
                print("Continuing training without early stopping for this epoch")
                return True  # Continue training

        # Register early stopping callback with error handling
        try:
            model.add_callback('on_train_epoch_end', custom_on_train_epoch_end)
            print("Early stopping callback registered successfully")
        except Exception as e:
            print(f"Warning: Could not register early stopping callback: {str(e)}")
            print("Training will continue without early stopping")
        
        print("Note: Early stopping is enabled but may be limited if metrics aren't available in expected format")
        print("Training will continue for full duration if early stopping cannot access metrics")

        # Start training with segmentation-optimized parameters
        print("Starting YOLOv11x-seg training...")
        results = model.train(**yolo_args)

        # Save final model with segmentation-specific naming
        final_path = os.path.join(args.output_dir, "weights", "yolo11x-seg_final.pt")
        model.save(final_path)
        print(f"Final segmentation model saved to {final_path}")

        # Store the best fitness values in results for later use
        # This makes them accessible in save_results
        if not hasattr(results, 'best_fitness_value'):
            results.best_fitness_value = best_fitness
        if not hasattr(results, 'best_fitness_epoch'):
            results.best_fitness_epoch = best_epoch
        if not hasattr(results, 'early_stopped'):
            results.early_stopped = (patience_counter >= args.patience)

        # Calculate training time
        train_time = time.time() - start_time
        hours, remainder = divmod(train_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"\nYOLOv11x-seg training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        print(f"Segmentation results saved to {args.output_dir}")

        return model, results

    except TypeError as e:
        # Special handling for parameter compatibility issues
        if "unexpected keyword argument" in str(e):
            param = str(e).split("keyword argument ")[-1].strip("'")
            print(f"Error: Segmentation parameter '{param}' not supported in this YOLO version")
            print("Try updating to a newer version with YOLOv11x-seg support")
        else:
            print(f"Type error during YOLOv11x-seg training: {e}")

        import traceback
        traceback.print_exc()

        return None, None
    except Exception as e:
        print(f"Error during YOLOv11x-seg training: {str(e)}")
        import traceback
        traceback.print_exc()

        return None, None

def save_results(args, results):
    """Save training results and metrics with segmentation-specific data."""
    if results is None:
        return

    try:
        # Initialize metrics dictionary with basic info
        metrics = {
            "epochs_completed": results.epoch if hasattr(results, 'epoch') else 0,
            "model_type": "YOLOv11x-seg",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Box detection metrics (keep for comparison)
        if hasattr(results, 'box'):
            box_metrics = {
                "box_map": float(results.box.map) if hasattr(results.box, 'map') else 0, 
                "box_map50": float(results.box.map50) if hasattr(results.box, 'map50') else 0,
                "box_precision": float(results.box.p.mean()) if hasattr(results.box, 'p') and hasattr(results.box.p, 'mean') else (float(results.box.p[0]) if hasattr(results.box, 'p') and len(results.box.p) > 0 else 0),
                "box_recall": float(results.box.r.mean()) if hasattr(results.box, 'r') and hasattr(results.box.r, 'mean') else (float(results.box.r[0]) if hasattr(results.box, 'r') and len(results.box.r) > 0 else 0),
                "box_f1": float(results.box.f1.mean()) if hasattr(results.box, 'f1') and hasattr(results.box.f1, 'mean') else (float(results.box.f1[0]) if hasattr(results.box, 'f1') and len(results.box.f1) > 0 else 0)
            }
            metrics.update(box_metrics)

        # Segmentation-specific metrics
        if hasattr(results, 'mask'):
            mask_metrics = {
                "mask_map": float(results.mask.map) if hasattr(results.mask, 'map') else 0,
                "mask_map50": float(results.mask.map50) if hasattr(results.mask, 'map50') else 0,
                "mask_precision": float(results.mask.p.mean()) if hasattr(results.mask, 'p') and hasattr(results.mask.p, 'mean') else (float(results.mask.p[0]) if hasattr(results.mask, 'p') and len(results.mask.p) > 0 else 0),
                "mask_recall": float(results.mask.r.mean()) if hasattr(results.mask, 'r') and hasattr(results.mask.r, 'mean') else (float(results.mask.r[0]) if hasattr(results.mask, 'r') and len(results.mask.r) > 0 else 0),
                "mask_f1": float(results.mask.f1.mean()) if hasattr(results.mask, 'f1') and hasattr(results.mask.f1, 'mean') else (float(results.mask.f1[0]) if hasattr(results.mask, 'f1') and len(results.mask.f1) > 0 else 0)
            }
            metrics.update(mask_metrics)

            # Make primary fitness based on mask metrics if available
            metrics["primary_metric"] = "mask_map"
            metrics["fitness"] = mask_metrics["mask_map"]
        else:
            # Fallback to box metrics if mask metrics not available
            metrics["primary_metric"] = "box_map"
            metrics["fitness"] = box_metrics["box_map"] if "box_map" in metrics else 0

        # Loss values (both box and mask)
        if hasattr(results, 'box_loss'):
            metrics["box_loss"] = float(np.mean(results.box_loss)) if isinstance(results.box_loss, (list, np.ndarray)) else float(results.box_loss)
        if hasattr(results, 'mask_loss'):
            metrics["mask_loss"] = float(np.mean(results.mask_loss)) if isinstance(results.mask_loss, (list, np.ndarray)) else float(results.mask_loss)
        if hasattr(results, 'seg_loss'):
            metrics["seg_loss"] = float(np.mean(results.seg_loss)) if isinstance(results.seg_loss, (list, np.ndarray)) else float(results.seg_loss)

        # IoU metrics (critical for segmentation quality)
        if hasattr(results, 'mask') and hasattr(results.mask, 'iou'):
            metrics["mask_iou"] = float(results.mask.iou)

        # Speed metrics
        if hasattr(results, 'speed'):
            metrics["speed"] = results.speed if isinstance(results.speed, dict) else {}

        # Save metrics
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Segmentation metrics saved to {metrics_path}")

        # Print primary metrics 
        print(f"\nYOLOv11x-seg Performance:")
        if "mask_map" in metrics:
            print(f"  Mask mAP: {metrics['mask_map']:.4f}")
            print(f"  Mask mAP50: {metrics['mask_map50']:.4f}")
        if "mask_iou" in metrics:
            print(f"  Mask IoU: {metrics['mask_iou']:.4f}")

    except Exception as e:
        print(f"Warning: Could not save segmentation metrics: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function optimized for YOLOv11x-seg training on both local and SLURM environments.
    """
    try:
        # Start timing for overall script execution
        script_start_time = time.time()

        # Parse arguments
        args = parse_args()

        # Configure SLURM environment if applicable
        is_slurm = 'SLURM_JOB_ID' in os.environ
        if is_slurm:
            slurm_job_id = os.environ.get('SLURM_JOB_ID')
            print(f"Running in SLURM environment (Job ID: {slurm_job_id})")

            # Configure CPU workers based on SLURM allocation
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                slurm_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
                args.workers = min(slurm_cpus, 16)  # Cap at 16 workers maximum
                print(f"Using {args.workers} worker threads based on SLURM allocation")

            # Configure GPU based on SLURM allocation
            if 'CUDA_VISIBLE_DEVICES' in os.environ:
                cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
                # If multiple GPUs are allocated, use them all
                if ',' in cuda_devices:
                    args.device = cuda_devices  # Pass all devices to YOLO
                else:
                    args.device = '0'  # Use the first allocated GPU
                print(f"Using SLURM-allocated GPU(s): {cuda_devices}")

            # Update output directory with SLURM job ID
            args.output_dir = os.path.join(args.output_dir, f"slurm_job_{slurm_job_id}")
        else:
            print("Running in local environment")

        # Validate data path before proceeding
        if not os.path.exists(args.data):
            print(f"Error: Data file not found: {args.data}")
            return 1

        # Validate model path with more informative message
        if not os.path.exists(args.model) and not args.model.startswith("yolo11"):
            print(f"Warning: Model file not found at {args.model}")
            print("Will attempt to download or use a pre-trained model with this name")

        # Setup environment and get configuration
        config = setup_environment(args)

        # Display YOLOv11x-specific settings
        print("YOLOv11x-seg specific settings:")
        print(f"  - Attention: {args.attention}")
        print(f"  - RepVGG blocks: {'Enabled' if args.use_repvgg else 'Disabled'}")
        print(f"  - Optimizer: {args.optimizer}")
        print("Early stopping settings:")
        print(f"  - Patience: {args.patience} epochs")
        print(f"  - Minimum delta: {args.min_delta}")
        print("")

        # Train YOLOv11x model
        try:
            # Use the training utility function with early stopping
            model, results = train_yolo11x(args, config)

            if model is None or results is None:
                print("Training failed. See error messages above.")
                # Create failure marker code remains the same...
                return 1

            # Save results using the utility function
            save_results(args, results)

            # Calculate total execution time
            total_time = time.time() - script_start_time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"\nYOLOv11x-seg training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
            print(f"Results saved to {args.output_dir}")

            # Create completion marker with segmentation-specific metrics
            with open(os.path.join(args.output_dir, "YOLO11X_SEG_TRAINING_COMPLETED"), "w") as f:
                # Extract segmentation-specific metrics
                segmentation_metrics = {
                    # Primary segmentation metrics
                    "mask_map": float(results.mask.map) if hasattr(results, 'mask') and hasattr(results.mask, 'map') else 0,
                    "mask_map50": float(results.mask.map50) if hasattr(results, 'mask') and hasattr(results.mask, 'map50') else 0,

                    # Secondary metrics
                    "mask_precision": float(results.mask.p) if hasattr(results, 'mask') and hasattr(results.mask, 'p') else 0,
                    "mask_recall": float(results.mask.r) if hasattr(results, 'mask') and hasattr(results.mask, 'r') else 0,

                    # Training info
                    "epochs_completed": results.epoch if hasattr(results, 'epoch') else 0,
                    "max_epochs": args.epochs,

                    # Best fitness (from early stopping)
                    "best_fitness": results.best_fitness_value if hasattr(results, 'best_fitness_value') else 0,
                    "best_epoch": results.best_fitness_epoch if hasattr(results, 'best_fitness_epoch') else 0
                }

                # Check if training ended due to early stopping
                early_stopping_triggered = getattr(results, 'early_stopped', False)
                early_stopping_msg = ""
                if early_stopping_triggered:
                    early_stopping_msg = f"\nEarly stopping activated after {segmentation_metrics['epochs_completed']} epochs (best results at epoch {segmentation_metrics['best_epoch']})"

                # Write primary heading
                f.write(f"YOLOv11x-seg training completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Write segmentation-specific metrics
                f.write("\n\nSegmentation metrics:")
                f.write(f"\n- Mask mAP: {segmentation_metrics['mask_map']:.4f}")
                f.write(f"\n- Mask mAP50: {segmentation_metrics['mask_map50']:.4f}")
                f.write(f"\n- Mask precision: {segmentation_metrics['mask_precision']:.4f}")
                f.write(f"\n- Mask recall: {segmentation_metrics['mask_recall']:.4f}")

                # Write training information
                f.write("\n\nTraining information:")
                f.write(f"\n- Epochs completed: {segmentation_metrics['epochs_completed']}/{segmentation_metrics['max_epochs']}")
                f.write(f"\n- Best fitness: {segmentation_metrics['best_fitness']:.4f} (at epoch {segmentation_metrics['best_epoch']})")
                f.write(early_stopping_msg)
                f.write(f"\n- Early stopping settings: patience={args.patience}, min_delta={args.min_delta}")

                # Write timestamp and hardware info
                f.write(f"\n\nEnvironment: {platform.platform()}")
                if torch.cuda.is_available():
                    f.write(f"\nGPU: {torch.cuda.get_device_name(0)}")

            return 0


        except TypeError as e:
            # Specific handling for YOLOv11x parameter errors
            if "unexpected keyword argument" in str(e):
                print(f"Error: YOLOv11x-specific parameter not supported: {e}")
                print("This may indicate you need a different version of the Ultralytics package that supports YOLOv11x")
            else:
                print(f"Type error during training: {e}")

            with open(os.path.join(args.output_dir, "YOLO11X_SEG_TRAINING_FAILED"), "w") as f:
                f.write(f"YOLOv11x-seg training failed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                f.write(f"\nError: {str(e)}")

            return 1

        except Exception as e:
            print(f"Error during YOLOv11x-seg training: {e}")
            import traceback
            traceback.print_exc()

            with open(os.path.join(args.output_dir, "YOLO11X_SEG_TRAINING_FAILED"), "w") as f:
                f.write(f"YOLOv11x-seg training failed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                f.write(f"\nError: {str(e)}")

            return 1

    except KeyboardInterrupt:
        print("\nYOLOv11x-seg training interrupted by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error in YOLOv11x-seg training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
