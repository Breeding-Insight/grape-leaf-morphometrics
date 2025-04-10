import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from scipy import ndimage
from skimage import measure


def enhance_mask_details(mask):
    """Simple edge detection for leaf serrations"""
    # Convert mask to binary
    binary_mask = mask > 0.5
    
    # Convert to uint8 format (0-255)
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    
    # Detect edges
    edges = cv2.Canny(mask_uint8, 50, 150)
    
    return binary_mask, edges

def get_instance_segmentation_model(num_classes):
    # Use original architecture settings
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Keep original dimensions (256)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256  
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      hidden_layer,
                                                      num_classes)
    return model

class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotations_file, transforms):
        self.root = root
        self.transforms = transforms
        self.coco = torchvision.datasets.CocoDetection(root, annotations_file)
        self.ids = list(sorted(self.coco.ids))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img, target = self.coco[index]

        boxes = []
        labels = []
        masks = []
        area = []
        iscrowd = []

        for annotation in target:
            bbox = annotation['bbox']
            x1, y1, width, height = bbox
            x2 = x1 + width
            y2 = y1 + height
            boxes.append([x1, y1, x2, y2])

            category_id = annotation['category_id']
            labels.append(category_id)

            if 'segmentation' in annotation:
                mask = self.coco.coco.annToMask(annotation)
                masks.append(mask)

            if 'area' in annotation:
                area.append(annotation['area'])
            else:
                area.append(width * height)

            if 'iscrowd' in annotation:
                iscrowd.append(annotation['iscrowd'])
            else:
                iscrowd.append(0)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        if masks:
            masks = np.stack(masks)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
        else:
            masks = torch.zeros((0, img.height, img.width), dtype=torch.uint8)

        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['masks'] = masks
        target['image_id'] = torch.tensor([img_id])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform():
    transforms = []
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def visualize_detection(image, target, output, idx, save_dir='detection_results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Training annotations
    ax2.imshow(image)
    ax2.set_title('Training Annotation')
    ax2.axis('off')
    
    # Show ground truth masks (training annotations)
    if 'masks' in target:
        gt_masks = target['masks']
        for gt_mask in gt_masks:
            mask_array = gt_mask.numpy()
            ax2.contour(mask_array, colors='yellow', linewidths=1)
    
    # Model detections
    ax3.imshow(image)
    ax3.set_title('Model Detection')
    ax3.axis('off')
    
    pred_boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    
    colors = ['magenta', 'cyan', 'blue', 'purple', 'green']
    
    # Sort predictions by score
    indices = np.argsort(scores)[::-1]
    
    for idx in indices:
        score = scores[idx]
        # Using 0.9 as threshold
        if score > 0.9:  # Increased from 0.7 to 0.9
            mask = masks[idx][0]
            box = pred_boxes[idx]
            color = colors[idx % len(colors)]
            
            # Get mask and apply stricter threshold
            binary_mask = (mask > 0.9).astype(np.uint8)  # Also increased this threshold
            
            # Filter small masks
            if np.sum(binary_mask) < 100:
                continue
                
            # Find contours
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE
            )
            
            # Filter by contour area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    valid_contours.append(contour)
            
            if not valid_contours:
                continue
                
            # Draw bounding box
            x1, y1, x2, y2 = box
            area_percentage = ((x2-x1) * (y2-y1)) / (image.size[0] * image.size[1]) * 100
            
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax3.add_patch(rect)
            
            # Add color overlay
            mask_overlay = np.zeros((*binary_mask.shape, 4))
            mask_overlay[binary_mask > 0] = (*plt.cm.colors.to_rgb(color), 0.2)
            ax3.imshow(mask_overlay)
            
            # Draw contours
            for contour in valid_contours:
                contour = contour.squeeze()
                if len(contour.shape) > 1:
                    ax3.plot(contour[:, 0], contour[:, 1], 
                            color='white', linewidth=2, alpha=0.8)
            
            ax3.text(
                x1, y1-5,
                f'Score: {score:.2f}\nArea: {area_percentage:.2f}%',
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=8,
                color='black'
            )
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'detection_{idx}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()
    print(f"Saved detection visualization to {save_path}")

def examine_annotations(test_dataloader):
    # Get first image and its annotations
    images, targets = next(iter(test_dataloader))
    
    print("\nChecking annotation details:")
    print(f"Number of images: {len(images)}")
    print("\nTarget contents:")
    for key in targets[0].keys():
        print(f"- {key}: {type(targets[0][key])}")
    
    # Check mask details
    if 'masks' in targets[0]:
        masks = targets[0]['masks']
        print(f"\nMask information:")
        print(f"Mask shape: {masks.shape}")
        print(f"Mask values range: {masks.min().item()} to {masks.max().item()}")

def evaluate(model, test_dataloader, device):
    model.eval()
    coco_gt = test_dataloader.dataset.coco.coco
    results = []
    
    print("\nProcessing predictions...")
    for idx, (images, targets) in enumerate(test_dataloader):
        print(f"\nProcessing image {idx+1}")
        
        images = list(img.to(device) for img in images)
        
        with torch.no_grad():
            outputs = model(images)

        for img, target, output in zip(images, targets, outputs):
            visualize_detection(T.ToPILImage()(img.cpu()), target, output, idx)

        for output, target in zip(outputs, targets):
            image_id = target['image_id'].item()
            boxes = output['boxes'].cpu()
            scores = output['scores'].cpu()

            for box, score in zip(boxes, scores):
                if score < 0.5:
                    continue
                xmin, ymin, xmax, ymax = box.tolist()
                results.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'score': score.item()
                })

    print(f"\nTotal predictions: {len(results)}")
    
    if not results:
        print("No detections found!")
        return

    coco_dt = coco_gt.loadRes(results)
    cocoEval = COCOeval(coco_gt, coco_dt, 'bbox')
    cocoEval.params.iouThrs = np.array([0.5, 0.75, 0.95])
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("\nExplicit IoU Threshold Results:")
    print(f"mAP50: {cocoEval.stats[1]:.3f}")
    print(f"mAP75: {cocoEval.stats[2]:.3f}")
    print(f"mAP95: {cocoEval.stats[3]:.3f}")

def main():
    test_dir = "/workdir/maw396/grape_leaves/annotations/test"
    test_annotations = "/workdir/maw396/grape_leaves/annotations/test/_annotations.coco.json"
    model_path = "/workdir/data/grape/grape_pheno/grape_leaf_metrics/models/mask_rcnn/checkpoints_20250319_161910/maskrcnn_model_final.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    print("\nModel Information:")
    print(f"Number of classes: {num_classes}")
    
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model loaded successfully")

    test_dataset = LeafDataset(test_dir, test_annotations, get_transform())
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    print(f"Test dataset size: {len(test_dataset)} images")

    print("\nStarting evaluation...")
    evaluate(model, test_dataloader, device)

if __name__ == "__main__":
    main()
