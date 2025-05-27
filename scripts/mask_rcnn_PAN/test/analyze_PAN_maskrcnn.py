import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def plot_precision_recall_curve(precisions, recalls, ap, output_dir):
    """Plot precision-recall curve"""
    plt.figure(figsize=(10, 8))
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.title(f'Precision-Recall Curve (AP: {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    save_path = os.path.join(output_dir, 'precision_recall_curve.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved precision-recall curve to {save_path}")


def analyze_iou_thresholds(coco_gt, coco_dt, total_gt_objects, output_dir):
    """Analyze detection performance at different IoU thresholds"""
    # List of IoU thresholds to evaluate
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    
    results = []
    
    # Create directory for IoU analysis plots
    iou_plots_dir = os.path.join(output_dir, 'iou_analysis')
    os.makedirs(iou_plots_dir, exist_ok=True)
    
    print("\nDetection counts at different IoU thresholds:")
    print("IoU     | True Positives | False Positives | False Negatives | "
          "Precision | Recall")
    print("--------+----------------+-----------------+-----------------+"
          "-----------+-------")
    # Store data for plotting
    plot_data = {
        'iou': [],
        'precision': [],
        'recall': [],
        'tp': [],
        'fp': [],
        'fn': []
    }
    
    # Evaluate at each IoU threshold
    for iou in iou_thresholds:
        # Create a new COCOeval object for this specific IoU threshold
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
        cocoEval.params.iouThrs = [iou]  # Set just this one IoU threshold
        
        # Run evaluation
        cocoEval.evaluate()
        cocoEval.accumulate()
        
        # Get results for this IoU
        precision = cocoEval.eval['precision']
        # precision has shape [iouThreshold, recThreshold, class, area, maxDets]
        # We want precision for all classes, all areas, and all max detections
        # We're using IoU threshold = 0 since we set only one
        # We want the maximum precision at each recall threshold
        precision_at_iou = precision[0, :, 0, 0, 2]  # For class 0, area 0, max dets 2
        recall_at_iou = cocoEval.params.recThrs
        
        # Calculate AP at this IoU
        ap_at_iou = np.mean(precision_at_iou[precision_at_iou > -1]) \
            if len(precision_at_iou[precision_at_iou > -1]) > 0 else 0
        
        # Get evaluation counts
        # This is a safer approach than trying to access 'dtMatches' which may not exist
        # Instead, we calculate from the precision and recall values
        
        # Count total detections (at score threshold 0.5)
        detections = [ann for ann in coco_dt.dataset['annotations']
                      if ann['score'] >= 0.5]
        total_detections = len(detections)
        
        # Calculate metrics
        if precision_at_iou[-1] > 0 and recall_at_iou[-1] > 0:
            final_precision = precision_at_iou[-1]
            final_recall = recall_at_iou[-1]
            
            # Estimate TP, FP, FN
            true_positives = int(final_recall * total_gt_objects)
            false_negatives = total_gt_objects - true_positives
            
            if final_precision > 0:
                false_positives = int(true_positives * (1 - final_precision) / final_precision)
            else:
                false_positives = total_detections
        else:
            # If precision or recall is 0, set defaults
            true_positives = 0
            false_positives = total_detections
            false_negatives = total_gt_objects
            final_precision = 0.0
            final_recall = 0.0
        
        # Print results
        print(f"IoU={iou:.2f} | {true_positives:14d} | {false_positives:15d} | "
              f"{false_negatives:15d} | {final_precision:.3f}   | {final_recall:.3f}")
        
        # Store in results
        results.append({
            'iou_threshold': iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': final_precision,
            'recall': final_recall,
            'ap': ap_at_iou
        })
        
        # Store for plotting
        plot_data['iou'].append(iou)
        plot_data['precision'].append(final_precision)
        plot_data['recall'].append(final_recall)
        plot_data['tp'].append(true_positives)
        plot_data['fp'].append(false_positives)
        plot_data['fn'].append(false_negatives)
        
        # Create precision-recall curve for this IoU
        if len(precision_at_iou[precision_at_iou > -1]) > 0:
            plot_precision_recall_curve(
                precision_at_iou[precision_at_iou > -1], 
                recall_at_iou[precision_at_iou > -1],
                ap_at_iou,
                iou_plots_dir
            )
    
    # Save detailed results
    with open(os.path.join(output_dir, 'iou_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot trends across IoU thresholds
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot precision and recall vs IoU
    axes[0, 0].plot(plot_data['iou'], plot_data['precision'], 'b-',
                    linewidth=2, marker='o')
    axes[0, 0].set_title('Precision vs IoU Threshold')
    axes[0, 0].set_xlabel('IoU Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(plot_data['iou'], plot_data['recall'], 'r-', linewidth=2, marker='o')
    axes[0, 1].set_title('Recall vs IoU Threshold')
    axes[0, 1].set_xlabel('IoU Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True)
    
    # Plot TP, FP, FN vs IoU
    axes[1, 0].plot(plot_data['iou'], plot_data['tp'], 'g-', linewidth=2, marker='o', label='True Positives')
    axes[1, 0].plot(plot_data['iou'], plot_data['fp'], 'r-', linewidth=2, marker='s', label='False Positives')
    axes[1, 0].set_title('TP/FP vs IoU Threshold')
    axes[1, 0].set_xlabel('IoU Threshold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(plot_data['iou'], plot_data['fn'], 'k-', linewidth=2, marker='d', label='False Negatives')
    axes[1, 1].set_title('FN vs IoU Threshold')
    axes[1, 1].set_xlabel('IoU Threshold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_threshold_analysis.png'), dpi=300)
    plt.close()
    
    return results


def evaluate_coco(results_path, coco_gt_path):
    """Run COCO evaluation on saved results"""
    # Load COCO ground truth
    coco_gt = COCO()
    with open(coco_gt_path, 'r') as f:
        coco_gt.dataset = json.load(f)
    coco_gt.createIndex()
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    if not results:
        print("No detections found!")
        return None
    
    # Load results into COCO format
    coco_dt = coco_gt.loadRes(results)
    
    # Create evaluation object
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    return cocoEval, coco_gt, coco_dt


def main():
    parser = argparse.ArgumentParser(description='Analyze Mask R-CNN detection results')
    parser.add_argument('--results_dir', type=str, 
                        default='data/processed/mask-rcnn_PAN',
                        help='Directory containing processed results')
    args = parser.parse_args()
    
    # Paths
    results_dir = args.results_dir
    results_path = os.path.join(results_dir, 'coco_results.json')
    coco_gt_path = os.path.join(results_dir, 'coco_gt.json')
    summary_path = os.path.join(results_dir, 'summary.json')
    
    # Create analysis directory
    analysis_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Load summary stats
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    total_gt_objects = summary['total_gt_objects']
    total_detected_objects = summary['total_detected_objects']
    
    print(f"\nTotal ground truth objects: {total_gt_objects}")
    print(f"Total detected objects (score >= 0.5): {total_detected_objects}")
    
    # Run COCO evaluation
    print("\nRunning COCO evaluation...")
    cocoEval, coco_gt, coco_dt = evaluate_coco(results_path, coco_gt_path)
    
    if cocoEval is None:
        print("Evaluation failed. Check if results file exists and contains valid detections.")
        return
    
    print("\nDetailed Evaluation Results:")
    print(f"AP (IoU=0.50:0.95): {cocoEval.stats[0]:.3f}")  # Primary COCO metric - AP over all IoUs
    print(f"AP (IoU=0.50): {cocoEval.stats[1]:.3f}")
    print(f"AP (IoU=0.75): {cocoEval.stats[2]:.3f}")
    
    # Print and store AP at every IoU interval (for 'all' category)
    ap_per_iou = {}
    for i, iou in enumerate(cocoEval.params.iouThrs):
        # The precision array shape: [T, R, K, A, M]
        # T: IoU thresholds, R: recall thresholds, K: categories, A: areas, M: max dets
        # For 'all' category, area=0, maxDets=2 (default), and mean over recall and categories
        precision = cocoEval.eval['precision'][i, :, :, 0, 2]
        valid = precision[precision > -1]
        ap = float(np.mean(valid)) if valid.size > 0 else float('nan')
        ap_per_iou[f"AP_{iou:.2f}"] = ap
        print(f"AP at IoU={iou:.2f}: {ap:.3f}")

    eval_stats = {
        'AP_all': float(cocoEval.stats[0]),
        'AP_50': float(cocoEval.stats[1]),
        'AP_75': float(cocoEval.stats[2]),
        'total_gt_objects': total_gt_objects,
        'total_detected_objects': total_detected_objects,
        'AP_per_IoU': ap_per_iou
    }
    
    with open(os.path.join(analysis_dir, 'coco_eval_stats.json'), 'w') as f:
        json.dump(eval_stats, f, indent=2)
    
    # Analyze at different IoU thresholds
    print("\nAnalyzing performance at different IoU thresholds...")
    analyze_iou_thresholds(coco_gt, coco_dt, total_gt_objects, analysis_dir)
    
    print(f"\nAnalysis complete! Results saved to {analysis_dir}")


if __name__ == "__main__":
    main() 