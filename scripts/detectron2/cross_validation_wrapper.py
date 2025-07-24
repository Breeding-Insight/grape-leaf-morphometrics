#!/usr/bin/env python3
"""
Cross-Validation Wrapper for Detectron2 Training

This script implements k-fold cross-validation for object detection/segmentation
training using Detectron2. It splits the dataset into k folds, trains k models,
and provides robust performance estimates.

Usage:
    python cross_validation_wrapper.py --config config.yaml --k_folds 5 --full_dataset path/to/full_dataset.json
"""

import os
import logging
import json
import argparse
import subprocess
import yaml
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from typing import Dict, List, Tuple, Any
import shutil
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Any


def load_coco_dataset(json_path: str) -> Dict[str, Any]:
    """Load COCO format dataset."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_coco_subset(full_data: Dict[str, Any], image_ids: List[int], img_dirs: List[str], retry_attempts: int = 3, min_valid_fraction: float = 0.8) -> Dict[str, Any]:
    """
    Create a subset of COCO data with only existing images and their annotations.
    Verifies file existence across multiple possible directories to prevent "file not found" skips during training.

    Args:
        full_data: Complete COCO dataset
        image_ids: List of image IDs to include
        img_dirs: List of base image directories from config to search for images.
        retry_attempts: Number of retries for file existence checks (for flaky FS)
        min_valid_fraction: Fail if fewer than this fraction of images are valid
        
    Returns:
        Validated subset of COCO data with reassigned annotation IDs
    """
    image_id_set = set(image_ids)
    subset_images = []
    skipped_images = []
    total_expected = len(image_ids)

    for img in full_data['images']:
        if img['id'] not in image_id_set:
            continue
        
        original_file_name = img['file_name']
        
        # Always normalize to basename. The training script will prepend the correct path.
        file_name = os.path.basename(original_file_name)
        
        # Find the full path by checking all provided image directories with retries
        exists = False
        potential_path = None # Define to be accessible after the loop
        for attempt in range(retry_attempts):
            for base_dir in img_dirs:
                potential_path = os.path.join(base_dir, file_name)
                if os.path.exists(potential_path) and os.path.isfile(potential_path):
                    exists = True
                    break  # Found it, break from inner loop
            if exists:
                break  # Found it, break from outer retry loop
            
            logging.warning(f"Attempt {attempt+1}/{retry_attempts}: File not found in any directory: {file_name} (original: {original_file_name})")
            if attempt < retry_attempts - 1:
                time.sleep(1)  # Delay for FS sync, but not on the last attempt
        
        if not exists:
            skipped_images.append((img['id'], original_file_name, file_name))
            continue
        
        # Store with normalized basename for Detectron2
        new_img = img.copy()
        new_img['file_name'] = file_name
        subset_images.append(new_img)

    # Check threshold
    valid_count = len(subset_images)
    if valid_count < total_expected * min_valid_fraction:
        raise ValueError(f"Too many invalid images ({len(skipped_images)} skipped out of {total_expected}). "
                         f"Check logs for details. Skipped: {skipped_images}")
    
    if skipped_images:
        logging.info(f"Subset created: {valid_count} valid images ({len(skipped_images)} skipped). Skipped details: {skipped_images}")
    else:
        logging.info(f"Subset created: {valid_count} valid images (0 skipped)")
    
    # Filter annotations for valid images only
    valid_image_ids = {img['id'] for img in subset_images}
    subset_annotations = [ann for ann in full_data['annotations'] if ann['image_id'] in valid_image_ids]
    
    # Reassign annotation IDs
    for i, ann in enumerate(subset_annotations, start=1):
        ann['id'] = i
    
    return {
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': full_data['categories'],
        'info': full_data.get('info', {}),
        'licenses': full_data.get('licenses', [])
    }


def save_coco_json(data: Dict[str, Any], filepath: str):
    """Save COCO format data to JSON file."""
    # Validate annotation IDs are unique
    if 'annotations' in data:
        ann_ids = [ann['id'] for ann in data['annotations']]
        if len(set(ann_ids)) != len(ann_ids):
            print(f"Warning: Duplicate annotation IDs found in {filepath}")
            print(f"Total annotations: {len(ann_ids)}, Unique IDs: {len(set(ann_ids))}")
            # Reassign IDs to ensure uniqueness
            for i, annotation in enumerate(data['annotations']):
                annotation['id'] = i
            print(f"Fixed: Reassigned annotation IDs to ensure uniqueness")
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def create_fold_config(base_config: Dict[str, Any], fold_idx: int, 
                      train_json: str, val_json: str, output_dir: str, image_dirs: List[str]) -> Dict[str, Any]:
    """
    Create a configuration for a specific fold.
    
    Args:
        base_config: Base configuration dictionary
        fold_idx: Current fold index
        train_json: Path to training JSON file
        val_json: Path to validation JSON file
        output_dir: Base output directory
        image_dirs: List of directories to search for images
        
    Returns:
        Modified configuration for this fold
    """
    # Deep copy the base config
    fold_config = yaml.safe_load(yaml.dump(base_config))
    
    # Update dataset paths
    fold_config['datasets']['train_json'] = train_json
    fold_config['datasets']['val_json'] = val_json
    fold_config['datasets']['image_dirs'] = image_dirs  # Pass the full list of searchable dirs
    
    # Update output directory
    fold_config['output_dir'] = os.path.join(output_dir, f"fold_{fold_idx}")
    
    return fold_config


def save_fold_config(config: Dict[str, Any], filepath: str):
    """Save configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def parse_evaluation_metrics(output_dir: str) -> Dict[str, float]:
    """
    Parse evaluation metrics from the training output.
    
    Args:
        output_dir: Directory containing training results
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Look for metrics in common locations
    possible_metric_files = [
        os.path.join(output_dir, "metrics.json"),
        os.path.join(output_dir, "coco_instances_results.json"),
        os.path.join(output_dir, "inference", "coco_instances_results.json")
    ]
    
    for metric_file in possible_metric_files:
        if os.path.exists(metric_file):
            try:
                with open(metric_file, 'r') as f:
                    content = f.read().strip()
                    
                    # Check if file is empty
                    if not content:
                        continue
                    
                    # Try to parse as standard JSON first
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict):
                            metrics.update(data)
                        elif isinstance(data, list) and len(data) > 0:
                            # Handle list format - take the last entry (most recent metrics)
                            metrics.update(data[-1])
                        break
                    except json.JSONDecodeError:
                        # Handle line-by-line JSON objects (common with training logs)
                        print(f"Attempting to parse line-by-line JSON from {metric_file}")
                        
                        # Split into lines and try to parse each as separate JSON
                        lines = content.strip().split('\n')
                        json_objects = []
                        
                        for line_num, line in enumerate(lines, 1):
                            line = line.strip()
                            if line:  # Skip empty lines
                                try:
                                    obj = json.loads(line)
                                    json_objects.append(obj)
                                except json.JSONDecodeError as line_error:
                                    print(f"Warning: Could not parse line {line_num} in {metric_file}: {line_error}")
                                    continue
                        
                        if json_objects:
                            # Take the last valid JSON object (most recent metrics)
                            metrics.update(json_objects[-1])
                            
                            # Also fix the file by writing it as a proper JSON array
                            try:
                                with open(metric_file, 'w') as fix_f:
                                    json.dump(json_objects, fix_f, indent=2)
                                print(f"Fixed malformed JSON in {metric_file} - converted to proper JSON array")
                            except Exception as fix_error:
                                print(f"Warning: Could not fix {metric_file}: {fix_error}")
                            
                            break
                        else:
                            print(f"Warning: No valid JSON objects found in {metric_file}")
                            continue
                            
            except (IOError, OSError) as e:
                print(f"Warning: Could not read metrics from {metric_file}: {e}")
                continue
    
    return metrics


def run_training_fold(config_path: str, fold_idx: int, resume: bool = False) -> bool:
    """
    Run training for a single fold.
    
    Args:
        config_path: Path to the fold-specific config file
        fold_idx: Current fold index
        resume: Whether to resume training
        
    Returns:
        True if training completed successfully, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Starting training for Fold {fold_idx + 1}")
    print(f"{'='*60}")
    
    # Build command - use the training script in the X101-FPN directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(script_dir, "X101-FPN", "train_x101-fpn.py")
    cmd = [
        "python", train_script_path,
        "--config", config_path
    ]
    
    if resume:
        cmd.append("--resume")
    
    # Run training
    try:
        # By removing capture_output=True, the subprocess output will be streamed in real-time.
        subprocess.run(cmd, text=True, check=True)
        print(f"Training completed successfully for Fold {fold_idx + 1}")
        return True
    except subprocess.CalledProcessError:
        # The error from the subprocess will have already been printed to stderr.
        print(f"Training failed for Fold {fold_idx + 1}")
        return False


def calculate_cv_statistics(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Calculate cross-validation statistics.
    
    Args:
        metrics_list: List of metrics dictionaries from each fold
        
    Returns:
        Dictionary containing mean, std, and individual fold results
    """
    if not metrics_list:
        return {}
    
    # Get all unique metric keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())
    
    cv_stats = {
        'individual_folds': metrics_list,
        'summary': {}
    }
    
    # Calculate statistics for each metric
    for key in all_keys:
        values = []
        for metrics in metrics_list:
            if key in metrics and isinstance(metrics[key], (int, float)):
                values.append(metrics[key])
        
        if values:
            cv_stats['summary'][key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    return cv_stats


def save_cv_results(cv_stats: Dict[str, Any], output_dir: str):
    """Save cross-validation results to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"cv_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(cv_stats, f, indent=2, default=str)
    
    # Save summary
    summary_file = os.path.join(output_dir, f"cv_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("Cross-Validation Results Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for metric, stats in cv_stats['summary'].items():
            f.write(f"{metric}:\n")
            f.write(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write(f"  Individual values: {[f'{v:.4f}' for v in stats['values']]}\n\n")
    
    print(f"\nCross-validation results saved to:")
    print(f"  Detailed: {results_file}")
    print(f"  Summary: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Validation Training for Detectron2")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to base YAML config file")
    parser.add_argument("--full_dataset", type=str, required=True,
                       help="Path to full COCO dataset JSON file")
    parser.add_argument("--k_folds", type=int, default=5,
                       help="Number of folds for cross-validation")
    parser.add_argument("--output_dir", type=str, default="cv_results",
                       help="Base output directory for all folds")
    parser.add_argument("--resume", action="store_true",
                       help="Resume training for existing folds")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducible splits")
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Resolve image directories from config and ensure they exist
    project_root = Path(__file__).resolve().parents[3]
    
    # Get image directories from config (supports both old and new formats)
    if 'image_dirs' in base_config['datasets']:
        # New format with image_dirs list
        image_search_dirs = []
        for img_dir in base_config['datasets']['image_dirs']:
            full_path = os.path.join(project_root, img_dir)
            if os.path.isdir(full_path):
                image_search_dirs.append(full_path)
            else:
                print(f"Warning: Image directory not found: {full_path}")
        
        if not image_search_dirs:
            raise ValueError("No valid image directories found in config")
            
    else:
        # Legacy format with separate train_img_dir and val_img_dir
        train_img_dir = os.path.join(project_root, base_config['datasets']['train_img_dir'])
        val_img_dir = os.path.join(project_root, base_config['datasets']['val_img_dir'])
        
        if not os.path.isdir(train_img_dir):
            raise ValueError(f"Training image directory not found: {train_img_dir}")
        if not os.path.isdir(val_img_dir):
            raise ValueError(f"Validation image directory not found: {val_img_dir}")
            
        image_search_dirs = [train_img_dir, val_img_dir]

    # Load full dataset
    print(f"Loading dataset from: {args.full_dataset}")
    full_data = load_coco_dataset(args.full_dataset)
    
    # Get image IDs for splitting
    image_ids = [img['id'] for img in full_data['images']]
    print(f"Total images: {len(image_ids)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup cross-validation splits
    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.random_state)
    
    # Store metrics from all folds
    all_metrics = []
    
    # Create temporary directory for fold data
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Run cross-validation
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_ids)):
            print(f"\n{'='*60}")
            print(f"Processing Fold {fold_idx + 1}/{args.k_folds}")
            print(f"{'='*60}")
            
            # Get image IDs for this fold
            train_image_ids = [image_ids[i] for i in train_indices]
            val_image_ids = [image_ids[i] for i in val_indices]
            
            print(f"Train images: {len(train_image_ids)}")
            print(f"Val images: {len(val_image_ids)}")
            
            # Create data subsets, searching in both original train and val directories
            train_subset = create_coco_subset(full_data, train_image_ids, image_search_dirs)
            val_subset = create_coco_subset(full_data, val_image_ids, image_search_dirs)
            
            # Save temporary JSON files
            train_json_path = os.path.join(temp_dir, f"train_fold_{fold_idx}.json")
            val_json_path = os.path.join(temp_dir, f"val_fold_{fold_idx}.json")
            
            save_coco_json(train_subset, train_json_path)
            save_coco_json(val_subset, val_json_path)
            
            
            # Create fold-specific config
            fold_config = create_fold_config(
                base_config, fold_idx, train_json_path, val_json_path, args.output_dir, image_search_dirs
            )
            
            # Save fold config
            fold_config_path = os.path.join(temp_dir, f"config_fold_{fold_idx}.yaml")
            save_fold_config(fold_config, fold_config_path)
            
            # Run training for this fold
            success = run_training_fold(fold_config_path, fold_idx, args.resume)
            
            if success:
                # Parse metrics
                fold_output_dir = fold_config['output_dir']
                metrics = parse_evaluation_metrics(fold_output_dir)
                
                if metrics:
                    all_metrics.append(metrics)
                    print(f"Fold {fold_idx + 1} metrics: {metrics}")
                else:
                    print(f"Warning: No metrics found for Fold {fold_idx + 1}")
            else:
                print(f"Warning: Training failed for Fold {fold_idx + 1}")
    
    # Calculate and save cross-validation statistics
    if all_metrics:
        cv_stats = calculate_cv_statistics(all_metrics)
        save_cv_results(cv_stats, args.output_dir)
        
        # Print summary
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        for metric, stats in cv_stats['summary'].items():
            print(f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    else:
        print("No successful training runs to analyze.")


if __name__ == "__main__":
    main() 