import json
import numpy as np
from pathlib import Path
from collections import Counter
import argparse

def analyze_coco_annotations(ann_path, show_plots=False, k_anchors=5):
    # Load COCO annotations
    with open(ann_path, 'r') as f:
        coco = json.load(f)

    widths, heights, areas, aspect_ratios = [], [], [], []
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        widths.append(w)
        heights.append(h)
        areas.append(w * h)
        aspect_ratios.append(w / h if h > 0 else 0)

    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)

    print(f"Total objects: {len(widths)}")
    print(f"Width  (px): min={widths.min():.1f}, 25%={np.percentile(widths,25):.1f}, median={np.median(widths):.1f}, 75%={np.percentile(widths,75):.1f}, max={widths.max():.1f}")
    print(f"Height (px): min={heights.min():.1f}, 25%={np.percentile(heights,25):.1f}, median={np.median(heights):.1f}, 75%={np.percentile(heights,75):.1f}, max={heights.max():.1f}")
    print(f"Area   (px²): min={areas.min():.1f}, 25%={np.percentile(areas,25):.1f}, median={np.median(areas):.1f}, 75%={np.percentile(areas,75):.1f}, max={areas.max():.1f}")
    print(f"Aspect ratio (w/h): 10%={np.percentile(aspect_ratios,10):.2f}, 50%={np.median(aspect_ratios):.2f}, 90%={np.percentile(aspect_ratios,90):.2f}")

    # K-means for anchor side lengths
    try:
        from sklearn.cluster import KMeans
        side_lengths = np.sqrt(areas).reshape(-1, 1)
        kmeans = KMeans(n_clusters=k_anchors, random_state=0).fit(side_lengths)
        anchor_sizes = sorted([int(x) for x in kmeans.cluster_centers_.flatten()])
        print(f"\nSuggested anchor side lengths (px): {anchor_sizes}")
    except ImportError:
        print("\nInstall scikit-learn for k-means anchor suggestion: pip install scikit-learn")
        anchor_sizes = None

    # Aspect ratio quantiles
    ar_quantiles = np.percentile(aspect_ratios, [10, 30, 50, 70, 90])
    ar_suggestions = [round(float(x), 2) for x in ar_quantiles]
    print(f"Suggested aspect ratios (10/30/50/70/90 percentiles): {ar_suggestions}")

    # Print RPN config suggestion
    if anchor_sizes:
        print("\nRPN anchor config suggestion for Mask R-CNN:")
        print("anchor_sizes = (")
        for s in anchor_sizes:
            print(f"    ({s},),")
        print(")")
        print(f"aspect_ratios = ({tuple(ar_suggestions)},) * len(anchor_sizes)")

    # Optional: plot histograms
    if show_plots:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,4))
        plt.subplot(131)
        plt.hist(np.sqrt(areas), bins=40)
        plt.title("√Area (side length)")
        plt.xlabel("Side length (px)")
        plt.subplot(132)
        plt.hist(widths, bins=40)
        plt.title("Width (px)")
        plt.subplot(133)
        plt.hist(aspect_ratios, bins=40)
        plt.title("Aspect ratio (w/h)")
        plt.xlabel("w/h")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_file", help="Path to COCO _annotations.coco.json")
    parser.add_argument("--plots", action="store_true", help="Show histograms")
    parser.add_argument("--k", type=int, default=5, help="Number of anchor sizes (k-means clusters)")
    args = parser.parse_args()
    analyze_coco_annotations(args.ann_file, show_plots=args.plots, k_anchors=args.k)