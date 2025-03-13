'''
OVERVIEW
Purpose: Train a Mask R-CNN model for instance segmentation on a custom leaf dataset.
Dataset: The dataset is loaded from COCO-formatted annotations.
Model: The model is initialized with pre-trained weights.
Training: The model is trained for a specified number of epochs.
Checkpoints: The script supports resuming training from checkpoints and saving checkpoints at each epoch.
Evaluation: The model is evaluated on validation and test sets.
Logging: Comprehensive logging and error handling are included.
Framework: The training is performed using the PyTorch framework.
Devices: The script is designed to run on various devices (CUDA, MPS, or CPU) based on availability.
Author: aja294@cornell.edu
'''

# Import standard libraries
import os
import time
from datetime import datetime
import gc
import traceback

# Import third-party libraries
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
from pycocotools.coco import COCO

# Import specific modules from torchvision
import torch.multiprocessing as mp
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Establish dataset class
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
    

# Define top level functions
## instance segmentation model
def get_instance_segmentation_model(num_classes):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


## transform function
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


## collate fn
def collate_fn(batch):
    return tuple(zip(*batch))


## Model eval
def evaluate_model(model, data_loader, device, log_message=print):
    """
    Evaluate the model on a dataset without computing gradients.
    Returns the average loss.
    """
    model.train()  # Temporarily set to train mode to compute losses
    total_loss = 0
    batch_count = 0

    try:
        with torch.no_grad():  # No gradients needed for evaluation
            for i, (images, targets) in enumerate(data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass and compute loss
                loss_dict = model(images, targets)

                # Sum all losses
                losses = sum(loss for loss in loss_dict.values())

                total_loss += losses.item()
                batch_count += 1

                if (i + 1) % 5 == 0:  # Log progress
                    log_message(f"  Eval batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}")

                # Clear memory
                del images, targets, loss_dict, losses
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        model.eval()  # Set back to eval mode
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    except Exception as e:
        model.eval()  # Ensure model is set back to eval mode even if an error occurs
        log_message(f"Error during evaluation: {e}")
        traceback.print_exc()
        return float('inf')
    

def main():
    # Set the path to your annotated images and annotations
    data_path = "/Users/aja294/Documents/Grape_local/projects/leaf_morphometrics/data/annotations"
    checkpoint_path = "/Users/aja294/Documents/Grape_local/projects/leaf_morphometrics/models/mask_rcnn"

    # Define directories for train, validation, and test
    train_dir = os.path.join(data_path, "coco", "train")
    val_dir = os.path.join(data_path, "coco", "valid")
    test_dir = os.path.join(data_path, "coco", "test")

    # Define annotation files
    train_annotations_file = os.path.join(train_dir, "_annotations.coco.json")
    val_annotations_file = os.path.join(val_dir, "_annotations.coco.json")
    test_annotations_file = os.path.join(test_dir, "_annotations.coco.json")

     # Create directories if they don't exist
    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"Data path: {data_path}")
    print(f"Train path: {train_dir}")
    print(f"Validation path: {val_dir}")
    print(f"Test path: {test_dir}")

    # Check if annotation files exist
    missing_files = []
    for file_path, name in [(train_annotations_file, "Training"), 
                            (val_annotations_file, "Validation"), 
                            (test_annotations_file, "Testing")]:
        if not os.path.exists(file_path):
            missing_files.append((name, file_path))

    if missing_files:
        for name, path in missing_files:
            print(f"Warning: {name} annotations file not found at {path}")
        print("Please ensure all annotation files exist before proceeding.")

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(checkpoint_path, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create log file
    log_file = os.path.join(checkpoint_dir, "training_log.txt")

    def log_message(message):
        """Write message to log file and print to console"""
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

    log_message(f"=== Training started at {timestamp} ===")

    # Load the datasets
    log_message("Loading datasets...")

    # Training dataset
    train_dataset = LeafDataset(train_dir, train_annotations_file, get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=16,
        shuffle=True,
        num_workers=2,  # Try with just 1 worker
        collate_fn=collate_fn,
        persistent_workers=True
    )
    log_message(f"Training dataset loaded with {len(train_dataset)} images")

    # Validation dataset
    val_dataset = None
    val_loader = None
    if os.path.exists(val_annotations_file):
        val_dataset = LeafDataset(val_dir, val_annotations_file, get_transform(train=False))
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        log_message(f"Validation dataset loaded with {len(val_dataset)} images")
    else:
        log_message("Validation dataset not available")

    # Test dataset
    test_dataset = None
    test_loader = None
    if os.path.exists(test_annotations_file):
        test_dataset = LeafDataset(test_dir, test_annotations_file, get_transform(train=False))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
        )
        log_message(f"Test dataset loaded with {len(test_dataset)} images")
    else:
        log_message("Test dataset not available")

    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log_message(f"Using device: {device}")

    # Get number of classes from training dataset
    num_classes = len(train_dataset.coco.coco.getCatIds()) + 1  # +1 for background
    log_message(f"Number of classes: {num_classes}")

    # Initialize model
    log_message("Initializing model...")
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    log_message("Model initialized and moved to device")

    # Check if there are existing checkpoints to resume from
    start_epoch = 0
    best_val_loss = float('inf')

    # Look for existing checkpoints in data_path/checkpoints
    base_checkpoint_dir = os.path.join(data_path, "checkpoints")
    if os.path.exists(base_checkpoint_dir):
        checkpoint_dirs = [d for d in os.listdir(base_checkpoint_dir) 
                            if os.path.isdir(os.path.join(base_checkpoint_dir, d))]

        if checkpoint_dirs:
            # Find the latest checkpoint directory
            latest_dir = max(checkpoint_dirs)
            latest_checkpoint_dir = os.path.join(base_checkpoint_dir, latest_dir)

            # Find the latest checkpoint file
            checkpoints = [f for f in os.listdir(latest_checkpoint_dir) 
                            if f.endswith('.pth') and f.startswith('mask_rcnn_checkpoint_epoch_')]

            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(latest_checkpoint_dir, latest_checkpoint)
                log_message(f"Loading checkpoint from {checkpoint_path}")

                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    log_message(f"Resuming from epoch {start_epoch}")
                except Exception as e:
                    log_message(f"Error loading checkpoint: {e}")
                    log_message("Starting training from scratch")
                    start_epoch = 0
    
    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Load optimizer and scheduler states if resuming
    if start_epoch > 0:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_message("Loaded optimizer and scheduler states")
        except Exception as e:
            log_message(f"Error loading optimizer/scheduler states: {e}")

    log_message("Optimizer and learning rate scheduler initialized")

    # Training parameters
    num_epochs = 5
    log_message(f"Starting training for {num_epochs} epochs")
    log_message(f"Using device: {device}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_loss = 0
        batch_count = 0

        log_message(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            batch_count += 1

            if (i + 1) % 10 == 0:  # Print every 10 batches
                log_message(f"  Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")

        # Print epoch training summary
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        log_message(f"  Epoch {epoch+1} training completed. Average Loss: {avg_train_loss:.4f}")

        # Validation phase
        # Validation phase
        val_loss = None
        if val_loader:
            val_loss = evaluate_model(model, val_loader, device, log_message)
            log_message(f"  Validation Loss: {val_loss:.4f}")

        # Update learning rate
        lr_scheduler.step()
        log_message(f"  Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'num_classes': num_classes
        }

        checkpoint_path = os.path.join(checkpoint_dir, f'mask_rcnn_checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, checkpoint_path)
        log_message(f"  Checkpoint saved to {checkpoint_path}")

        # Save best model based on validation loss
        if val_loss is not None and (epoch == start_epoch or val_loss < best_val_loss):
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'mask_rcnn_best_model.pth')
            torch.save(checkpoint, best_model_path)
            log_message(f"  New best model saved to {best_model_path} (val_loss: {best_val_loss:.4f})")

        # Calculate epoch duration
        epoch_duration = time.time() - epoch_start_time
        log_message(f"  Epoch duration: {epoch_duration:.2f} seconds")

    log_message("Training completed!")

    # Evaluate on test set if available
    if test_loader:
        log_message("Evaluating on test set...")
        test_loss = evaluate_model(model, test_loader, device)
        log_message(f"Test Loss: {test_loss:.4f}")

      # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "maskrcnn_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'train_loss': avg_train_loss,
        'val_loss': val_loss if val_loader else None,
        'test_loss': test_loss if test_loader else None
    }, final_model_path)
    log_message(f"Final model saved to {final_model_path}")

    # Create a symlink to the latest checkpoint directory
    # Extract the base directory (without the checkpoints_timestamp part)
    base_dir = os.path.dirname(checkpoint_dir)
    latest_link = os.path.join(base_dir, "latest")

    # Check if the symlink already exists and remove it
    if os.path.exists(latest_link):
        try:
            os.remove(latest_link)
        except:
            # On some systems, we might need to use unlink for symlinks
            os.unlink(latest_link)

    # Create the symlink pointing to the current checkpoint directory
    try:
        os.symlink(checkpoint_dir, latest_link)
        log_message(f"Created symlink to latest checkpoint directory: {latest_link}")
    except Exception as e:
        log_message(f"Error creating symlink: {e}")

if __name__ == "__main__":
    main()