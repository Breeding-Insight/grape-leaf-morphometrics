"""
YOLO Model Testing Script
=========================

This script provides a comprehensive testing framework for YOLOv11-seg models, optimized for Apple 
Silicon (M1/M2) Macs with MPS acceleration, but also supporting CUDA and CPU environments.

Features:
- Automatically detects and utilizes appropriate hardware acceleration (MPS, CUDA, or CPU)
- Comprehensive model validation with detailed metrics reporting
- Generates visual outputs including confusion matrices and detection visualizations
- Memory-optimized for resource-constrained environments
- Structured output organization with JSON metrics and summary reports

Requirements:
- Python 3.8+
- PyTorch with MPS/CUDA support (as available)
- Ultralytics YOLO package
- Trained YOLO model (.pt format)
- Data configuration (data.yaml)

Usage:
    python test_yolo.py --model [MODEL_PATH] --data [DATA_YAML] --output-dir [OUTPUT_DIR]

The script handles various test scenarios and provides detailed performance metrics suitable
for local testing before deployment on high-performance computing environments.

Author: Arlyn Ackerman
Contact: aja294@cornell.edu
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
import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Numerical operations

# YOLO library
from ultralytics import YOLO  # YOLO model from Ultralytics

# Define top level functions===================================================
# Utility Functions
# Define command line arguments


def parse_args():
    """Parse command line arguments for model testing."""
    parser = argparse.ArgumentParser(description='Test trained YOLOv11-seg model locally on M1 Mac')

    # Required arguments
    parser.add_argument('--model', type=str,
                        default='yolo11n-seg.pt',
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--data', type=str,
                        default='yolo_versions/metrics_marker_detection.v3i.yolov11/data.yaml',
                        help='Path to test data.yaml file')

    # Output arguments
    parser.add_argument('--output-dir', type=str, 
                        default='./runs/detect/test_local',
                        help='Output directory for test results')

    # Test parameters
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for testing (smaller for M1)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for testing')
    parser.add_argument('--device', type=str, 
                        default='mps' if torch.backends.mps.is_available() else 'cpu',
                        help='Device to test on (mps for M1, cpu as fallback)')
    parser.add_argument('--workers', type=int, default=2,
                        help='Number of worker threads for data loading')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')

    # Output options
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results to *.txt file')
    parser.add_argument('--save-conf', action='store_true',
                        help='Save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true',
                        help='Save results to COCO json format')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped prediction boxes')
    parser.add_argument('--hide-labels', action='store_true',
                        help='Hide labels in output visualizations')
    parser.add_argument('--hide-conf', action='store_true',
                        help='Hide confidences in output visualizations')

    # Local testing specific
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode with additional logging')

    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories for outputs."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def check_environment():
    """Check and print environment information."""
    env_info = {
        "Platform": platform.platform(),
        "Python": platform.python_version(),
        "PyTorch": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "MPS Available": torch.backends.mps.is_available(),
        "Device": "MPS" if torch.backends.mps.is_available() else 
                 "CUDA" if torch.cuda.is_available() else "CPU"
    }

    print("\n=== Environment Information ===")
    for k, v in env_info.items():
        print(f"{k}: {v}")
    print("")

    return env_info

### Core Functionality
def plot_confusion_matrix(results, output_dir):
    """Generate and save confusion matrix plot."""
    try:
        # Get confusion matrix from results
        conf_matrix = results.confusion_matrix.matrix

        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()

        # Add class labels if available
        classes = results.names
        if classes:
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")
        return True
    except (AttributeError, TypeError) as e:
        print(f"Could not generate confusion matrix: {e}")
        return False

def test_model(model_path, args):
    """Test the model and save results."""
    print(f"Testing model: {model_path}")
    start_time = time.time()

    # Load model
    try:
        model = YOLO(model_path)
        print(f"Model loaded successfully: {type(model)}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Clear cache if using MPS or CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache emptied")
    elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        torch.mps.empty_cache()
        print("MPS cache emptied")
    else:
        print("No GPU cache to empty (running on CPU)")

    # Run testing with error handling
    try:
        print(f"Starting validation on device: {args.device}")
        results = model.val(
            data=args.data,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            workers=args.workers,
            conf=args.conf_thres,
            iou=args.iou_thres,
            plots=True,  # Generate test plots
            save_json=args.save_json,
            save_conf=args.save_conf,
            save_txt=args.save_txt,
            save_crop=args.save_crop,
            project=os.path.dirname(args.output_dir),
            name=os.path.basename(args.output_dir),
            exist_ok=True,
            verbose=args.verbose
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Save test metrics
        metrics_path = os.path.join(args.output_dir, "test_metrics.json")

        # Safely extract metrics with fallbacks
        metrics = {
            "mAP50-95": float(getattr(results.box, 'map', 0)),
            "mAP50": float(getattr(results.box, 'map50', 0)),
            "precision": float(getattr(results.box, 'p', 0)),
            "recall": float(getattr(results.box, 'r', 0)),
            "f1-score": float(getattr(results.box, 'f1', 0)),
            "speed": {
                "preprocess": float(results.speed.get('preprocess', 0)),
                "inference": float(results.speed.get('inference', 0)),
                "postprocess": float(results.speed.get('postprocess', 0))
            },
            "total_time_seconds": elapsed_time,
            "images_per_second": float(len(getattr(results.box, 'conf', [])) / elapsed_time if hasattr(results.box, 'conf') else 0)
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Test metrics saved to {metrics_path}")

        # Generate confusion matrix
        plot_confusion_matrix(results, args.output_dir)

        # Clear cache again after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()

        return results

    except Exception as e:
        print(f"Error during testing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None
    
### Post Processing
def create_summary(results, args, env_info):
    """Create a summary of test results."""
    if results is None:
        print("Cannot create summary: No results available")
        return

    summary_path = os.path.join(args.output_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write("=== YOLOv11-seg Testing Summary (Local M1 Mac) ===\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Test data: {args.data}\n")
        f.write(f"Completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== Environment ===\n")
        for k, v in env_info.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")

        f.write("=== Performance Metrics ===\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
        f.write(f"mAP50: {results.box.map50:.4f}\n")
        f.write(f"Precision: {results.box.p:.4f}\n")
        f.write(f"Recall: {results.box.r:.4f}\n")
        f.write(f"F1-Score: {results.box.f1:.4f}\n\n")

        f.write("=== Speed ===\n")
        f.write(f"Preprocess: {results.speed['preprocess']:.2f} ms/image\n")
        f.write(f"Inference: {results.speed['inference']:.2f} ms/image\n")
        f.write(f"Postprocess: {results.speed['postprocess']:.2f} ms/image\n")

        # Add M1-specific notes
        f.write("\n=== Notes ===\n")
        f.write("This test was run locally on an M1 Mac. Performance on the HPC server will likely be different.\n")
        f.write("Consider these results as a preliminary check before submitting to the server.\n")

    print(f"Summary saved to {summary_path}")

def main():
    """
    Main function for model testing - optimized for both local execution and SLURM jobs.
    """
    try:
        # Start timing for overall script execution
        script_start_time = time.time()

        # Parse arguments
        args = parse_args()

        # Check if running in SLURM environment
        is_slurm = 'SLURM_JOB_ID' in os.environ

        # Collect environment info
        if is_slurm:
            print("Detected SLURM environment")
            slurm_job_id = os.environ.get('SLURM_JOB_ID')

            # Adjust workers based on SLURM allocation
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                args.workers = min(int(os.environ['SLURM_CPUS_PER_TASK']), args.workers)
                print(f"Adjusted workers to {args.workers} based on SLURM allocation")

            # Auto-select GPU if available in SLURM
            if 'CUDA_VISIBLE_DEVICES' in os.environ and torch.cuda.is_available():
                cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
                if cuda_devices:
                    args.device = f'cuda:{cuda_devices.split(",")[0]}'
                    print(f"Using SLURM-assigned GPU: {args.device}")

            # Add job ID to output directory for organization
            orig_output_dir = args.output_dir
            args.output_dir = os.path.join(args.output_dir, f"job_{slurm_job_id}")
            print(f"Output directory updated: {orig_output_dir} → {args.output_dir}")
        else:
            print("Running in local environment")

        # Validate required files exist
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return 1

        if not os.path.exists(args.data):
            print(f"Error: Data file not found: {args.data}")
            return 1

        # Check environment
        env_info = check_environment()

        # Add SLURM info to environment details if applicable
        if is_slurm:
            slurm_vars = {k: v for k, v in os.environ.items() if k.startswith('SLURM')}
            env_info.update({"SLURM_VARS": slurm_vars})

        # Setup directories
        output_dir = setup_directories(args.output_dir)

        # Save command line arguments
        try:
            args_path = os.path.join(output_dir, "test_args.json")
            with open(args_path, "w") as f:
                json.dump(vars(args), f, indent=4)

            # Save environment info
            env_path = os.path.join(output_dir, "environment.json")
            with open(env_path, "w") as f:
                json.dump(env_info, f, indent=4)
        except IOError as e:
            print(f"Warning: Could not save configuration files: {e}")

        print(f"Output directory: {output_dir}")
        print(f"Test data path: {args.data}")
        print(f"Device: {args.device}")

        # Test model
        print("\n=== Starting model testing ===")
        results = test_model(args.model, args)

        if results is not None:
            # Create summary
            create_summary(results, args, env_info)

            # Calculate and show total execution time
            total_time = time.time() - script_start_time
            print(f"\nTesting complete in {total_time:.2f} seconds")
            print(f"Results saved to {output_dir}")

            # Create a completion marker (useful for both local and SLURM)
            with open(os.path.join(output_dir, "COMPLETED"), "w") as f:
                f.write(f"Completed successfully at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            return 0
        else:
            print("\nTesting failed. Check the error messages above.")
            with open(os.path.join(output_dir, "FAILED"), "w") as f:
                f.write(f"Testing failed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            return 1

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if 'args' in locals() and hasattr(args, 'debug') and args.debug:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

