# Standard library imports
import os
import multiprocessing as mp
import argparse
import traceback
import re
import concurrent.futures
import tempfile
import time

# Third-party libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T

# Local application/library specific imports
from plantcv import plantcv as pcv

# Set CUDA memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
def get_instance_segmentation_model(num_classes):
    """Initialize Mask R-CNN model"""
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def load_model(model_path):
    """Load Mask R-CNN model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 3  # Background + Leaf + Quarter
    model = get_instance_segmentation_model(num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device)
def detect_objects(image_path, model, threshold=0.98):  # High confidence threshold
    """Detect objects using Mask R-CNN model"""
    device = next(model.parameters()).device
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    # Filter predictions by threshold
    keep = prediction['scores'] > threshold
    boxes = prediction['boxes'][keep]
    scores = prediction['scores'][keep]
    masks = prediction['masks'][keep]
    
    predictions = []
    for box, score, mask in zip(boxes, scores, masks):
        box = box.cpu().numpy()
        center_x = int(box[0] + (box[2] - box[0])/2)
        center_y = int(box[1] + (box[3] - box[1])/2)
        width = int(box[2] - box[0])
        height = int(box[3] - box[1])
        
        binary_mask = mask[0].cpu().numpy() > 0.5
        area_px = np.sum(binary_mask)
        
        pred = {
            'x': center_x,
            'y': center_y,
            'width': width,
            'height': height,
            'confidence': float(score),
            'area_px': area_px,
            'points': [
                {'x': int(box[0]), 'y': int(box[1])},
                {'x': int(box[2]), 'y': int(box[1])},
                {'x': int(box[2]), 'y': int(box[3])},
                {'x': int(box[0]), 'y': int(box[3])}
            ]
        }
        predictions.append(pred)
    
    return {'predictions': predictions}

def detect_quarters(image_path, model):
    """Detect quarters using Mask R-CNN model"""
    return detect_objects(image_path, model, threshold=0.7)  # Lower threshold for quarters

def detect_leaves(image_path, model):
    """Detect leaves using Mask R-CNN model"""
    return detect_objects(image_path, model, threshold=0.9)  # Higher threshold for precise leaf detection
def get_input_images(input_dir):
    """Get list of input images"""
    try:
        if not os.path.isdir(input_dir):
            raise ValueError(f"Directory does not exist: {input_dir}")
        
        # List of files to exclude
        exclude_files = [
            'leaf_area_distribution',
            '1f389',
            'X-Large tray size frame',
            'Large tray size frame',
            'enhanced_reference'
        ]
        
        # Get only valid sample images
        valid_images = [
            os.path.join(input_dir, f) for f in os.listdir(input_dir) 
            if f.lower().endswith(('.tif', '.jpg', '.jpeg', '.png'))
            and not any(exclude in f for exclude in exclude_files)
        ]
        
        return valid_images
    except Exception as e:
        print(f"Error getting input images: {str(e)}")
        return []

def get_conversion_factor(folder_path, model):
    """Get pixel to mm conversion from reference image"""
    ref1 = cv2.imread(os.path.join(folder_path, "X-Large tray size frame.jpeg"), cv2.IMREAD_COLOR)
    if ref1 is None:
        raise ValueError("Failed to load reference image")

    print("Reference image loaded successfully")
    print(f"Image shape: {ref1.shape}")
    
    temp_path = os.path.join(tempfile.gettempdir(), f"temp_ref_{int(time.time())}.jpg")
    
    try:
        cv2.imwrite(temp_path, ref1)
        result = detect_quarters(temp_path, model)
        
        print("\nQuarter detection results:")
        print(f"Raw predictions: {result}")
        
        if 'predictions' in result and result['predictions']:
            detected_diameters = []
            detection_image = ref1.copy()
            
            for idx, pred in enumerate(result['predictions'], 1):
                print(f"\nQuarter {idx} confidence: {pred['confidence']:.3f}")
                
                center_x = int(pred['x'])
                center_y = int(pred['y'])
                width = int(pred['width'])
                height = int(pred['height'])
                detected_diameter = min(width, height)
                detected_diameters.append(detected_diameter)
                
                print(f"Quarter {idx} detection details:")
                print(f"Center: ({center_x}, {center_y})")
                print(f"Width: {width}px, Height: {height}px")
                print(f"Detected diameter: {detected_diameter}px")
                
                radius = int(detected_diameter/2)
                cv2.circle(detection_image, (center_x, center_y), radius, (0, 255, 0), 2)
                label = f"Quarter: {pred['confidence']:.2f}"
                cv2.putText(detection_image, label, 
                          (center_x - 40, center_y - radius - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
            
            if detected_diameters:
                avg_diameter = np.mean(detected_diameters)
                pix_per_mm = avg_diameter / 24.26
                
                print("\nConversion Factor Calculation:")
                print(f"Average diameter: {avg_diameter:.1f}px")
                print(f"Quarter actual diameter: 24.26mm")
                print(f"Conversion Factor: {pix_per_mm:.2f} pixels per mm")
                
                quarter_area_px = np.pi * (avg_diameter/2)**2
                quarter_area_mm2 = quarter_area_px / (pix_per_mm ** 2)
                print(f"\nVerification:")
                print(f"Quarter area: {quarter_area_mm2:.1f} mm² (should be ~462.8 mm²)")
                
                debug_path = os.path.join(folder_path, "quarter_detection_debug.jpg")
                cv2.imwrite(debug_path, detection_image)
                print(f"\nDebug image saved to: {debug_path}")
                
                return pix_per_mm
            else:
                raise ValueError("No quarters detected with sufficient confidence")
        else:
            print("Raw response:", result)
            raise ValueError("No quarters detected in reference image")
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
def determine_arrangement_type(leaf_detections, image_width, image_height):
    """Determine how leaves are arranged in the image"""
    if not leaf_detections:
        return "two_by_two"  # Default to two_by_two
    
    centers = [(pred['x'], pred['y']) for pred in leaf_detections]
    
    if len(centers) == 2:
        x_diff = abs(centers[0][0] - centers[1][0])
        y_diff = abs(centers[0][1] - centers[1][1])
        if x_diff > y_diff:
            return "horizontal_pair"
        else:
            return "vertical_pair"
    
    elif len(centers) == 4:
        sorted_by_y = sorted(centers, key=lambda c: c[1])
        y_diff = sorted_by_y[-1][1] - sorted_by_y[0][1]
        
        if y_diff > image_height/3:
            return "two_by_two"
        else:
            return "horizontal_four"
    
    return "two_by_two"

def label_leaf_position(x, y, arrangement_type, all_centers, leaf_index):
    """Label leaf position based on arrangement type"""
    if arrangement_type == "horizontal_pair":
        return "Left" if x < sorted([c[0] for c in all_centers])[1] else "Right"
    
    elif arrangement_type == "vertical_pair":
        return "Upper" if y < sorted([c[1] for c in all_centers])[1] else "Lower"
    
    elif arrangement_type == "two_by_two":
        sorted_y = sorted([c[1] for c in all_centers])
        mid_y = (sorted_y[1] + sorted_y[2]) / 2  # Use median for two_by_two
        sorted_x = sorted([c[0] for c in all_centers])
        mid_x = (sorted_x[1] + sorted_x[2]) / 2  # Use median for two_by_two
        
        if y < mid_y:
            return "Upper Left" if x < mid_x else "Upper Right"
        else:
            return "Bottom Left" if x < mid_x else "Bottom Right"
    
    elif arrangement_type == "horizontal_four":
        sorted_centers = sorted(all_centers, key=lambda c: c[0])
        positions = ["Far Left", "Center Left", "Center Right", "Far Right"]
        for i, (cx, cy) in enumerate(sorted_centers):
            if abs(x - cx) < 10 and abs(y - cy) < 10:
                return positions[i]


def process_single_image(args):
    """Process single image"""
    image_path, pix_per_mm, model = args
    
    MAX_LEAVES_PER_SAMPLE = 4
    QUARTER_AREA_CM2 = 4.628
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image_height, image_width = image.shape[:2]
        results = detect_leaves(image_path, model)
        
        all_measurements = []

        if 'predictions' in results:
            leaf_detections = results['predictions']
            arrangement_type = determine_arrangement_type(leaf_detections, image_width, image_height)
            all_centers = [(pred['x'], pred['y']) for pred in leaf_detections]
            
            for idx, prediction in enumerate(leaf_detections):
                center_x = prediction['x']
                center_y = prediction['y']
                position = label_leaf_position(center_x, center_y, arrangement_type, 
                                            all_centers, idx)
                
                leaf_area_px = prediction['area_px']
                leaf_area_mm2 = leaf_area_px / (pix_per_mm ** 2)
                leaf_area_cm2 = leaf_area_mm2 / 100
                quarter_ratio = leaf_area_cm2 / QUARTER_AREA_CM2

                measurement = {
                    'Sample_ID': file_name,
                    'Arrangement': arrangement_type,
                    'Leaf_Position': position,
                    'Area_px': round(leaf_area_px, 2),
                    'Area_mm2': round(leaf_area_mm2, 2),
                    'Area_cm2': round(leaf_area_cm2, 2),
                    'Confidence': prediction['confidence'],
                    'Quarter_Ratio': round(quarter_ratio, 8)
                }
                all_measurements.append(measurement)

            if all_measurements:
                df = pd.DataFrame(all_measurements)
                print(f"\nSummary for {file_name}:")
                print(f"Arrangement: {arrangement_type}")
                for pos in sorted(df['Leaf_Position'].unique()):
                    leaf_data = df[df['Leaf_Position'] == pos]
                    print(f"{pos}: {leaf_data['Area_cm2'].iloc[0]:.2f} cm²")
            
            return pd.DataFrame(all_measurements)

        return pd.DataFrame()

    except Exception as e:
        print(f"Error processing {file_name}: {str(e)}")
        return pd.DataFrame()

def analyze_leaf_distribution(df):
    """Analyze the distribution of leaf areas and add size categories"""
    areas = df['Area_cm2']
    
    quartiles = areas.quantile([0.25, 0.5, 0.75])
    
    def get_size_category(area):
        if area < quartiles[0.25]:
            return "Small"
        elif area < quartiles[0.5]:
            return "Medium"
        elif area < quartiles[0.75]:
            return "Large"
        else:
            return "Very Large"
    
    df['Size_Category'] = df['Area_cm2'].apply(get_size_category)
    
    plt.figure(figsize=(12, 6))
    plt.hist(areas, bins=50, edgecolor='black')
    plt.title('Distribution of Leaf Areas')
    plt.xlabel('Leaf Area (cm²)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    colors = ['r', 'g', 'b']
    for p, c in zip([0.25, 0.5, 0.75], colors):
        val = areas.quantile(p)
        plt.axvline(x=val, color=c, linestyle='--', 
                   label=f'{int(p*100)}th percentile: {val:.1f} cm²')
    
    plt.legend()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze leaf morphometrics from images")
    parser.add_argument('-i', '--input_dir', type=str, required=True,
                       help='Path to input directory of images')
    parser.add_argument('-m', '--model_path', type=str, 
                       default="/workdir/maw396/grape_leaves/models/mask_rcnn/checkpoints_20250328_172317/mask_rcnn_best_model.pth",
                       help='Path to Mask R-CNN model weights')
    parser.add_argument('-w', '--workers', type=int,
                       help='Number of worker processes')
    args = parser.parse_args()

    print("\nLoading Mask R-CNN model...")
    model = load_model(args.model_path)
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    print("\nGetting input images...")
    input_images = get_input_images(args.input_dir)
    if not input_images:
        print("No image files found in input directory")
        exit(1)
    print(f"Found {len(input_images)} images to process")

    print("\nProcessing reference image...")
    try:
        pix_per_mm = get_conversion_factor(args.input_dir, model)
        print(f"Calibration complete: {pix_per_mm:.2f} pixels per mm")
        
        # Clear CUDA cache after calibration
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error in calibration: {str(e)}")
        print(traceback.format_exc())
        exit(1)

    print("\nProcessing images...")
    results = []
    error_count = 0
    success_count = 0
    
    # Process images sequentially with progress bar and memory management
    for image_path in tqdm(input_images, desc="Processing images"):
        if "X-Large tray size frame.jpeg" not in image_path:
            try:
                # Clear CUDA cache before processing each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process image
                result = process_single_image((image_path, pix_per_mm, model))
                
                # Add result if valid
                if not result.empty:
                    results.append(result)
                    success_count += 1
                
            except Exception as e:
                print(f"\nError processing {image_path}: {str(e)}")
                error_count += 1

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df = analyze_leaf_distribution(final_df)
        
        output_path = os.path.join(args.input_dir, 'maskrcnn_leaf_measurements.csv')
        final_df.to_csv(output_path, index=False)
        
        dist_path = os.path.join(args.input_dir, 'maskrcnn_leaf_area_distribution.png')
        plt.savefig(dist_path)
        plt.close()
        
        print("\nProcessing Summary:")
        print(f"Total images processed successfully: {success_count}")
        print(f"Total images failed: {error_count}")
        print(f"Total leaves measured: {len(final_df)}")
        print(f"Results saved to: {output_path}")
        
        quartiles = final_df['Area_cm2'].quantile([0.25, 0.5, 0.75])
        print("\nLeaf Area Summary:")
        print(f"Mean area: {final_df['Area_cm2'].mean():.2f} cm²")
        print(f"Median area: {quartiles[0.5]:.2f} cm²")
        print(f"Min area: {final_df['Area_cm2'].min():.2f} cm²")
        print(f"Max area: {final_df['Area_cm2'].max():.2f} cm²")
        print(f"Standard deviation: {final_df['Area_cm2'].std():.2f} cm²")
        
        print("\nArrangement Distribution:")
        print(final_df['Arrangement'].value_counts())
    else:
        print("\nNo measurements were collected!")
        print(f"Total errors: {error_count}")

    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

