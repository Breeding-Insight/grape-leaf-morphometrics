import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def plot_precision_recall_curve(precisions, recalls, ap, output_dir):
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
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    results = []
    iou_plots_dir = os.path.join(output_dir, 'iou_analysis')
    os.makedirs(iou_plots_dir, exist_ok=True)
    print("\nDetection counts at different IoU thresholds:")
    print(
        "IoU     | True Positives | False Positives | False Negatives | "
        "Precision | Recall"
    )
    print(
        "--------+----------------+-----------------+-----------------+"
        "-----------+-------"
    )
    plot_data = {'iou': [], 'precision': [], 'recall': [], 'tp': [], 'fp': [], 'fn': []}
    for iou in iou_thresholds:
        cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
        cocoEval.params.iouThrs = [iou]
        cocoEval.evaluate()
        cocoEval.accumulate()
        precision = cocoEval.eval['precision']
        precision_at_iou = precision[0, :, 0, 0, 2]
        recall_at_iou = cocoEval.params.recThrs
        ap_at_iou = np.mean(precision_at_iou[precision_at_iou > -1]) \
            if len(precision_at_iou[precision_at_iou > -1]) > 0 else 0
        detections = [ann for ann in coco_dt.dataset['annotations']
                      if ann['score'] >= 0.5]
        total_detections = len(detections)
        if precision_at_iou[-1] > 0 and recall_at_iou[-1] > 0:
            final_precision = precision_at_iou[-1]
            final_recall = recall_at_iou[-1]
            true_positives = int(final_recall * total_gt_objects)
            false_negatives = total_gt_objects - true_positives
            if final_precision > 0:
                false_positives = int(
                    true_positives * (1 - final_precision) / final_precision
                )
            else:
                false_positives = total_detections
        else:
            true_positives = 0
            false_positives = total_detections
            false_negatives = total_gt_objects
            final_precision = 0.0
            final_recall = 0.0
        print(
            f"IoU={iou:.2f} | {true_positives:14d} | {false_positives:15d} | "
            f"{false_negatives:15d} | {final_precision:.3f}   | "
            f"{final_recall:.3f}"
        )
        results.append({
            'iou_threshold': iou,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': final_precision,
            'recall': final_recall,
            'ap': ap_at_iou
        })
        plot_data['iou'].append(iou)
        plot_data['precision'].append(final_precision)
        plot_data['recall'].append(final_recall)
        plot_data['tp'].append(true_positives)
        plot_data['fp'].append(false_positives)
        plot_data['fn'].append(false_negatives)
        if len(precision_at_iou[precision_at_iou > -1]) > 0:
            plot_precision_recall_curve(
                precision_at_iou[precision_at_iou > -1],
                recall_at_iou[precision_at_iou > -1],
                ap_at_iou,
                iou_plots_dir
            )
    with open(os.path.join(output_dir, 'iou_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    plt.figure(figsize=(12, 10))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].plot(plot_data['iou'], plot_data['precision'], 'b-',
                    linewidth=2, marker='o')
    axes[0, 0].set_title('Precision vs IoU Threshold')
    axes[0, 0].set_xlabel('IoU Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True)
    axes[0, 1].plot(plot_data['iou'], plot_data['recall'], 'r-',
                    linewidth=2, marker='o')
    axes[0, 1].set_title('Recall vs IoU Threshold')
    axes[0, 1].set_xlabel('IoU Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True)
    axes[1, 0].plot(plot_data['iou'], plot_data['tp'], 'g-',
                    linewidth=2, marker='o', label='True Positives')
    axes[1, 0].plot(plot_data['iou'], plot_data['fp'], 'r-',
                    linewidth=2, marker='s', label='False Positives')
    axes[1, 0].set_title('TP/FP vs IoU Threshold')
    axes[1, 0].set_xlabel('IoU Threshold')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 1].plot(plot_data['iou'], plot_data['fn'], 'k-',
                    linewidth=2, marker='d', label='False Negatives')
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
    coco_gt = COCO()
    with open(coco_gt_path, 'r') as f:
        coco_gt.dataset = json.load(f)
    coco_gt.createIndex()
    with open(results_path, 'r') as f:
        results = json.load(f)
    if not results:
        print("No detections found!")
        return None
    coco_dt = coco_gt.loadRes(results)
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval, coco_gt, coco_dt


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Detectron2 detection results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing processed results')
    args = parser.parse_args()
    results_dir = args.results_dir
    results_path = os.path.join(results_dir, 'coco_results.json')
    coco_gt_path = os.path.join(results_dir, 'coco_gt.json')
    summary_path = os.path.join(results_dir, 'summary.json')
    analysis_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    total_gt_objects = summary['total_gt_objects']
    total_detected_objects = summary['total_detected_objects']
    print(f"\nTotal ground truth objects: {total_gt_objects}")
    print(
        f"Total detected objects (score >= 0.5): "
        f"{total_detected_objects}"
    )
    print("\nRunning COCO evaluation...")
    cocoEval, coco_gt, coco_dt = evaluate_coco(results_path, coco_gt_path)
    if cocoEval is None:
        print("Evaluation failed. Check if results file exists and contains valid detections.")
        return
    print("\nDetailed Evaluation Results:")
    print(
        f"AP (IoU=0.50:0.95): {cocoEval.stats[0]:.3f}"
    )
    print(
        f"AP (IoU=0.50): {cocoEval.stats[1]:.3f}"
    )
    print(
        f"AP (IoU=0.75): {cocoEval.stats[2]:.3f}"
    )
    ap_per_iou = {}
    for i, iou in enumerate(cocoEval.params.iouThrs):
        precision = cocoEval.eval['precision'][i, :, :, 0, 2]
        valid = precision[precision > -1]
        ap = float(np.mean(valid)) if valid.size > 0 else float('nan')
        ap_per_iou[f"AP_{iou:.2f}"] = ap
        print(
            f"AP at IoU={iou:.2f}: {ap:.3f}"
        )
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
    print("\nAnalyzing performance at different IoU thresholds...")
    analyze_iou_thresholds(coco_gt, coco_dt, total_gt_objects, analysis_dir)
    print(
        f"\nAnalysis complete! Results saved to {analysis_dir}"
    )


if __name__ == "__main__":
    main() 