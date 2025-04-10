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
from ultralytics import YOLO

# Local application/library specific imports
from plantcv import plantcv as pcv

def load_model(model_path):
    """Load YOLO model"""
    return YOLO(model_path)

def detect_objects(image_path, model):
    """Detect objects using local YOLO model"""
    results = model(image_path)
    predictions = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            pred = {
                'x': int(box.xyxy[0][0] + (box.xyxy[0][2] - box.xyxy[0][0])/2),  # center x
                'y': int(box.xyxy[0][1] + (box.xyxy[0][3] - box.xyxy[0][1])/2),  # center y
                'width': int(box.xyxy[0][2] - box.xyxy[0][0]),  # width
                'height': int(box.xyxy[0][3] - box.xyxy[0][1]),  # height
                'confidence': float(box.conf),
                'class': result.names[int(box.cls)],
                'points': [  # Convert box to points for polygon
                    {'x': int(box.xyxy[0][0]), 'y': int(box.xyxy[0][1])},  # top-left
                    {'x': int(box.xyxy[0][2]), 'y': int(box.xyxy[0][1])},  # top-right
                    {'x': int(box.xyxy[0][2]), 'y': int(box.xyxy[0][3])},  # bottom-right
                    {'x': int(box.xyxy[0][0]), 'y': int(box.xyxy[0][3])}   # bottom-left
                ]
            }
            predictions.append(pred)
    
    return {'predictions': predictions}

def detect_quarters(image_path, model):
    """Detect quarters using local YOLO model"""
    return detect_objects(image_path, model)

def detect_leaves(image_path, model):
    """Detect leaves using local YOLO model"""
    return detect_objects(image_path, model)

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
        
        if 'predictions' in result and result['predictions']:
            detected_diameters = []
            detection_image = ref1.copy()
            
            for idx, pred in enumerate(result['predictions'], 1):
                center_x = int(pred['x'])
                center_y = int(pred['y'])
                width = int(pred['width'])
                height = int(pred['height'])
                detected_diameter = min(width, height)
                detected_diameters.append(detected_diameter)
                
                print(f"\nQuarter {idx} detection details:")
                print(f"Center: ({center_x}, {center_y})")
                print(f"Detected diameter: {detected_diameter}px")
                
                radius = int(detected_diameter/2)
                cv2.circle(detection_image, (center_x, center_y), radius, (0, 255, 0), 2)
                label = f"Quarter: {pred['confidence']:.2f}"
                cv2.putText(detection_image, label, 
                          (center_x - 40, center_y - radius - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
            
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
            
            plt.figure(figsize=(12,8))
            plt.title('Quarter Detections')
            plt.imshow(cv2.cvtColor(detection_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            
            return pix_per_mm
        else:
            print("Raw response:", result)
            raise ValueError("No quarters detected in reference image")
            
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
def determine_arrangement_type(leaf_detections, image_width, image_height):
    """Determine how leaves are arranged in the image"""
    if not leaf_detections:
        return "unknown"
    
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
    
    return "unknown"

def label_leaf_position(x, y, arrangement_type, all_centers, leaf_index):
    """Label leaf position based on arrangement type"""
    if arrangement_type == "horizontal_pair":
        return "Left" if x < sorted([c[0] for c in all_centers])[1] else "Right"
    
    elif arrangement_type == "vertical_pair":
        return "Upper" if y < sorted([c[1] for c in all_centers])[1] else "Lower"
    
    elif arrangement_type == "two_by_two":
        sorted_y = sorted([c[1] for c in all_centers])
        mid_y = (sorted_y[1] + sorted_y[2]) / 2
        sorted_x = sorted([c[0] for c in all_centers])
        mid_x = (sorted_x[1] + sorted_x[2]) / 2
        
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
    
    return f"Leaf_{leaf_index+1}"

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
        temp_path = os.path.join(tempfile.gettempdir(), 
                               f"temp_leaf_{os.getpid()}_{time.time_ns()}.jpg")
        try:
            cv2.imwrite(temp_path, image)
            results = detect_leaves(temp_path, model)
            
            all_measurements = []

            if 'predictions' in results:
                leaf_detections = [pred for pred in results['predictions'] if pred['class'] == 'Leaf']
                arrangement_type = determine_arrangement_type(leaf_detections, image_width, image_height)
                all_centers = [(pred['x'], pred['y']) for pred in leaf_detections]
                
                for idx, prediction in enumerate(leaf_detections):
                    center_x = prediction['x']
                    center_y = prediction['y']
                    position = label_leaf_position(center_x, center_y, arrangement_type, 
                                                all_centers, idx)
                    
                    points = np.array([[p['x'], p['y']] for p in prediction['points']], 
                                    dtype=np.int32).reshape((-1, 1, 2))
                    
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)
                    leaf_area_px = cv2.countNonZero(mask)
                    
                    leaf_area_mm2 = leaf_area_px / (pix_per_mm ** 2)
                    leaf_area_cm2 = leaf_area_mm2 / 100
                    quarter_ratio = leaf_area_cm2 / QUARTER_AREA_CM2

                    measurement = {
                        'area_cm2': leaf_area_cm2,
                        'leaf_area_px': leaf_area_px,
                        'leaf_area_mm2': leaf_area_mm2,
                        'quarter_ratio': quarter_ratio,
                        'confidence': prediction.get('confidence', 0),
                        'position': position,
                        'arrangement': arrangement_type
                    }
                    all_measurements.append(measurement)

                measurements = []
                for idx, meas in enumerate(all_measurements, 1):
                    measurements.append({
                        'Sample_ID': file_name,
                        'Arrangement': meas['arrangement'],
                        'Leaf_Position': meas['position'],
                        'Area_px': round(meas['leaf_area_px'], 2),
                        'Area_mm2': round(meas['leaf_area_mm2'], 2),
                        'Area_cm2': round(meas['area_cm2'], 2),
                        'Confidence': meas['confidence'],
                        'Quarter_Ratio': meas['quarter_ratio']
                    })

                if measurements:
                    df = pd.DataFrame(measurements)
                    print(f"\nSummary for {file_name}:")
                    print(f"Arrangement: {arrangement_type}")
                    for pos in sorted(df['Leaf_Position'].unique()):
                        leaf_data = df[df['Leaf_Position'] == pos]
                        print(f"{pos}: {leaf_data['Area_cm2'].iloc[0]:.2f} cm²")
                
                return pd.DataFrame(measurements)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

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
    parser.add_argument('-m', '--model_path', type=str, required=True,
                       help='Path to YOLO model weights')
    parser.add_argument('-w', '--workers', type=int,
                       help='Number of worker processes')
    args = parser.parse_args()

    print("\nLoading YOLO model...")
    model = load_model(args.model_path)

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
    except Exception as e:
        print(f"Error in calibration: {str(e)}")
        print(traceback.format_exc())
        exit(1)

    num_workers = args.workers or max(1, min(mp.cpu_count() - 1, 4))
    print(f"\nUsing {num_workers} workers for parallel processing")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for image_path in input_images:
            if "X-Large tray size frame.jpeg" not in image_path:
                futures.append(
                    executor.submit(process_single_image, (image_path, pix_per_mm, model))
                )

        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures),
                         desc='Processing images'):
            try:
                result = future.result()
                if not result.empty:
                    results.append(result)
            except Exception as e:
                print(f"\nException occurred: {e}")

    if results:
        final_df = pd.concat(results, ignore_index=True)
        final_df = analyze_leaf_distribution(final_df)
        
        output_path = os.path.join(args.input_dir, 'leaf_measurements.csv')
        final_df.to_csv(output_path, index=False)
        
        plt.savefig(os.path.join(args.input_dir, 'leaf_area_distribution.png'))
        plt.close()
        
        print("\nProcessing Summary:")
        print(f"Total images processed: {len(input_images)}")
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
