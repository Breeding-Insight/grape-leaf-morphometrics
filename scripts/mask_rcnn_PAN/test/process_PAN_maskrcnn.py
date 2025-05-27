import os
import torch
import torchvision
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import gc
import json

# Import the PANet model loader
from scripts.mask_rcnn_PAN.train_val.train_grape_mask_rcnn_PAN import get_mask_rcnn_panet_model

def enhance_mask_details(mask):
    """Simple edge detection for leaf serrations"""
    binary_mask = mask > 0.5
    mask_uint8 = binary_mask.astype(np.uint8) * 255
    edges = cv2.Canny(mask_uint8, 50, 150)
    return binary_mask, edges

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

def visualize_detection(image, target, output, idx, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 8))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    ax2.imshow(image)
    ax2.set_title('Training Annotation')
    ax2.axis('off')
    if 'masks' in target:
        gt_masks = target['masks']
        for gt_mask in gt_masks:
            mask_array = gt_mask.numpy()
            ax2.contour(mask_array, colors='yellow', linewidths=1)
    ax3.imshow(image)
    ax3.set_title('Model Detection')
    ax3.axis('off')
    pred_boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    masks = output['masks'].cpu().numpy()
    colors = ['magenta', 'cyan', 'blue', 'purple', 'green']
    indices = np.argsort(scores)[::-1]
    for idx in indices:
        score = scores[idx]
        if score > 0.9:
            mask = masks[idx][0]
            box = pred_boxes[idx]
            color = colors[idx % len(colors)]
            binary_mask = (mask > 0.9).astype(np.uint8)
            if np.sum(binary_mask) < 100:
                continue
            contours, _ = cv2.findContours(
                binary_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_NONE
            )
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    valid_contours.append(contour)
            if not valid_contours:
                continue
            x1, y1, x2, y2 = box
            area_percentage = ((x2-x1) * (y2-y1)) / (image.size[0] * image.size[1]) * 100
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax3.add_patch(rect)
            mask_overlay = np.zeros((*binary_mask.shape, 4))
            mask_overlay[binary_mask > 0] = (*plt.cm.colors.to_rgb(color), 0.2)
            ax3.imshow(mask_overlay)
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

def process_images(model, test_dataloader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    model.eval()
    coco_gt = test_dataloader.dataset.coco.coco
    results = []
    total_gt_objects = 0
    total_detected_objects = 0
    print("\nProcessing predictions...")
    for idx, (images, targets) in enumerate(test_dataloader):
        print(f"\nProcessing image {idx+1}")
        images = list(img.to(device) for img in images)
        with torch.no_grad():
            outputs = model(images)
        for img, target, output in zip(images, targets, outputs):
            image_id = target['image_id'].item()
            visualize_detection(T.ToPILImage()(img.cpu()), target, output, image_id, vis_dir)
            output_cpu = {k: v.cpu() for k, v in output.items()}
            del output
            gc.collect()
            torch.cuda.empty_cache()
            output_data = {
                'boxes': output_cpu['boxes'].numpy().tolist(),
                'scores': output_cpu['scores'].numpy().tolist(),
                'labels': output_cpu['labels'].numpy().tolist(),
                'image_id': image_id
            }
            gt_data = {
                'boxes': target['boxes'].cpu().numpy().tolist(),
                'labels': target['labels'].cpu().numpy().tolist(),
                'image_id': image_id
            }
            with open(os.path.join(output_dir, f'prediction_{image_id}.json'), 'w') as f:
                json.dump(output_data, f)
            with open(os.path.join(output_dir, f'ground_truth_{image_id}.json'), 'w') as f:
                json.dump(gt_data, f)
        for target in targets:
            if 'boxes' in target:
                total_gt_objects += len(target['boxes'])
        for output in outputs:
            scores = output['scores'].cpu()
            detected_in_image = sum(1 for score in scores if score >= 0.5)
            total_detected_objects += detected_in_image
            boxes = output['boxes'].cpu()
            for box, score in zip(boxes, scores):
                if score < 0.5:
                    continue
                xmin, ymin, xmax, ymax = box.tolist()
                results.append({
                    'image_id': target['image_id'].item(),
                    'category_id': 1,
                    'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
                    'score': score.item()
                })
        del img, target, output
        plt.close('all')
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nTotal ground truth objects: {total_gt_objects}")
    print(f"Total detected objects (score >= 0.5): {total_detected_objects}")
    with open(os.path.join(output_dir, 'coco_results.json'), 'w') as f:
        json.dump(results, f)
    with open(os.path.join(output_dir, 'coco_gt.json'), 'w') as f:
        json.dump(coco_gt.dataset, f)
    summary = {
        'total_gt_objects': total_gt_objects,
        'total_detected_objects': total_detected_objects
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f)
    print(f"All results saved to {output_dir}")

def main():
    test_dir = "data/annotations/coco100/test"
    test_annotations = "data/annotations/coco100/test/_annotations.coco.json"
    model_path = "checkpoints/mask-rcnn_PAN/checkpoints_20250506_213315/mask_rcnn_PANet_best_model.pth"
    output_dir = "data/processed/mask-rcnn_PAN"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Loading PANet checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    print("\nModel Information:")
    print(f"Number of classes: {num_classes}")
    model = get_mask_rcnn_panet_model(num_classes)
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
    print("\nStarting image processing...")
    process_images(model, test_dataloader, device, output_dir)

if __name__ == "__main__":
    main() 