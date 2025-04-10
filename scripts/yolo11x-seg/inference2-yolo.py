import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm

def load_test_annotations(image_file, test_dir):
    """Load ground truth annotations from test set"""
    # Get corresponding txt file path
    txt_file = os.path.splitext(image_file)[0] + '.txt'
    txt_path = os.path.join(test_dir, txt_file)
    
    annotations = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                # YOLO format: class x_center y_center width height
                values = line.strip().split()
                if len(values) == 5:
                    annotations.append([float(x) for x in values])
    return annotations

def visualize_detection(image, predictions, idx, test_dir, image_file, save_dir='yolo_detection_results'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nProcessing detection visualization for image {idx}")
    
    # Create figure with three panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Ground truth annotations
    ax2.imshow(image)
    ax2.set_title('Ground Truth Annotations')
    ax2.axis('off')
    
    # Load and draw ground truth annotations
    gt_annotations = load_test_annotations(image_file, test_dir)
    img_width, img_height = image.size
    for ann in gt_annotations:
        class_id, x_center, y_center, width, height = ann
        
        # Convert YOLO format to pixel coordinates
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        w = width * img_width
        h = height * img_height
        
        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor='yellow',
            facecolor='none'
        )
        ax2.add_patch(rect)
    
    # Model predictions
    ax3.imshow(image)
    ax3.set_title('Model Predictions')
    ax3.axis('off')
    
    if len(predictions) > 0:
        boxes = predictions[0].boxes
        masks = getattr(predictions[0], 'masks', None)
        print(f"Found {len(boxes)} raw detections")
        
        conf_vals = [float(box.conf) for box in boxes]
        sorted_indices = np.argsort(conf_vals)[::-1]
        
        colors = ['magenta', 'cyan', 'blue', 'purple', 'green']
        detections_count = 0
        
        for i in sorted_indices:
            box = boxes[i]
            confidence = float(box.conf)
            
            if confidence > 0.25:
                detections_count += 1
                print(f"Processing detection {detections_count} with confidence: {confidence:.2f}")
                
                # Draw mask if available
                if masks is not None:
                    mask = masks[i].data.cpu().numpy()[0]
                    mask_overlay = np.zeros((*mask.shape, 4))
                    mask_overlay[mask > 0.5] = (*plt.cm.colors.to_rgb(colors[detections_count % len(colors)]), 0.3)
                    ax3.imshow(mask_overlay)
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width = x2 - x1
                height = y2 - y1
                
                area_percentage = (width * height) / (image.size[0] * image.size[1]) * 100
                
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2,
                    edgecolor=colors[detections_count % len(colors)],
                    facecolor='none'
                )
                ax3.add_patch(rect)
                
                ax3.text(
                    x1, y1-5,
                    f'Score: {confidence:.2f}\nArea: {area_percentage:.2f}%',
                    bbox=dict(facecolor='white', alpha=0.8),
                    fontsize=8,
                    color='black'
               )

    save_path = os.path.join(save_dir, f'detection_{idx}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()
    return save_path  # Return the save path

def main():
    print("="*50)
    print("Starting YOLO inference script")
    print("="*50)
    
    # Clean up previous results
    if os.path.exists('yolo_detection_results'):
        import shutil
        shutil.rmtree('yolo_detection_results')
    print("Cleaned previous results directory")
    
    print("\nStep 1/4: Loading model...")
    model_path = "runs/segment/train6/weights/best.pt"
    print(f"Loading custom YOLO model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model at {model_path}")
    
    model = YOLO(model_path)
    print("Model loaded successfully!")
    
    print("\nStep 2/4: Setting up model configuration...")
    model.conf = 0.25
    model.iou = 0.45
    print(f"Model classes: {model.names}")
    print(f"Confidence threshold: {model.conf}")
    
    print("\nStep 3/4: Preparing test images...")
    test_dir = 'test/images'
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    total_images = len(image_files)
    print(f"Found {total_images} test images")
    
    print("\nStep 4/4: Processing images...")
    # Create progress bar for overall processing
    with tqdm(total=total_images, desc="Overall Progress") as pbar:
        for idx, image_file in enumerate(image_files):
            print(f"\n{'='*50}")
            print(f"Image {idx+1}/{total_images}: {image_file}")
            
            # Sub-steps for each image
            print("  → Loading image...", end=' ', flush=True)
            image_path = os.path.join(test_dir, image_file)
            image = Image.open(image_path)
            print("Done")
            
            print("  → Running inference...", end=' ', flush=True)
            results = model(image, verbose=True)
            print("Done")
            
            print("  → Creating visualization...", end=' ', flush=True)
            visualize_detection(image, results, idx, test_dir, image_file)  # Added test_dir and image_file parameters
            print("Done")
            
            # Update progress bar
            pbar.update(1)
            
            # Show detection statistics
            if len(results) > 0:
                boxes = results[0].boxes
                conf_scores = [float(box.conf) for box in boxes]
                if conf_scores:
                    print(f"  → Statistics:")
                    print(f"     - Detections: {len(conf_scores)}")
                    print(f"     - Max confidence: {max(conf_scores):.3f}")
                    print(f"     - Average confidence: {sum(conf_scores)/len(conf_scores):.3f}")

    print("\n" + "="*50)
    print("Processing completed!")
    print(f"Total images processed: {total_images}")
    print(f"Results saved in: yolo_detection_results/")
    print("="*50)

if __name__ == "__main__":
    main()
	
