'''
OVERVIEW
Purpose: Train a Mask R-CNN model for instance segmentation on a custom leaf dataset.
Dataset: The dataset is loaded from COCO-formatted annotations.
Model: The model is initialized with pre-trained weights.
Training: The model is trained for a specified number of epochs.
Checkpoints: The script supports resuming training from checkpoints and saving checkpoints at each epoch.
Evaluation: The model is evaluated on a validation set.
Logging: Comprehensive logging and error handling are included.
Framework: The training is performed using the PyTorch framework.
Devices: The script is optimized for metal but can run on various devices (CUDA, MPS, or CPU) based on availability.
Author: aja294@cornell.edu
'''

# Import standard libraries
import os
import gc
import time
from datetime import datetime
import gc
import traceback

# Import third-party libraries
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.amp import GradScaler, autocast

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
## Utility functions
### Transform
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

### Collate
def collate_fn(batch):
    return tuple(zip(*batch))

### Instance segmentation model
def get_instance_segmentation_model(num_classes):
    weights = torchvision.models.detection.MaskRCNN_ResNet101_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet101_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

### Create symlink
def create_symlink(checkpoint_dir, log_message):
    """
    Create symlink to latest checkpoint directory

    Args:
        checkpoint_dir (str): Path to the checkpoint directory to link to
        log_message (function): Function to log messages
    """
    # Extract the base directory (without the checkpoints_timestamp part)
    base_dir = os.path.dirname(checkpoint_dir)
    latest_link = os.path.join(base_dir, "latest")

    # Check if the symlink already exists and remove it
    if os.path.exists(latest_link):
        try:
            if os.path.islink(latest_link):
                # Use appropriate method based on platform
                if os.name == 'nt':  # Windows
                    os.remove(latest_link)
                else:  # Unix-like
                    os.unlink(latest_link)
            else:
                # It exists but is not a symlink (e.g., a directory)
                log_message(f"Warning: {latest_link} exists but is not a symlink. Removing it.")
                if os.path.isdir(latest_link):
                    import shutil
                    shutil.rmtree(latest_link)
                else:
                    os.remove(latest_link)

            log_message(f"Removed existing symlink: {latest_link}")
        except Exception as e:
            log_message(f"Error removing existing symlink: {e}")
            return  # Exit if we can't remove the existing symlink

    # Create the symlink pointing to the current checkpoint directory
    try:
        # Handle different platforms
        if os.name == 'nt':  # Windows
            # On Windows, we need admin privileges for symlinks or developer mode enabled
            # Use directory junction as an alternative if symlink fails
            try:
                os.symlink(checkpoint_dir, latest_link, target_is_directory=True)
            except:
                # If symlink fails, try using a directory junction (Windows-specific)
                import subprocess
                subprocess.check_call(['mklink', '/J', latest_link, checkpoint_dir], shell=True)
        else:  # Unix-like
            os.symlink(checkpoint_dir, latest_link)

        log_message(f"Created symlink to latest checkpoint directory: {latest_link} -> {checkpoint_dir}")
    except Exception as e:
        log_message(f"Error creating symlink: {e}")

## Define setup paths
def setup_paths_and_logging(base_path, models_path):
    """
    Setup paths, directories and logging infrastructure

    Args:
        base_path (str): Base path for data and annotations
        models_path (str): Path where model checkpoints will be stored

    Returns:
        tuple: Contains the following elements:
            - train_dir (str): Directory containing training data
            - val_dir (str): Directory containing validation data
            - train_annotations_file (str): Path to training annotations file
            - val_annotations_file (str): Path to validation annotations file
            - checkpoint_dir (str): Directory where checkpoints will be stored
            - log_message (function): Function to log messages to console and file
    """
    # Define directories for train and validation
    train_dir = os.path.join(base_path, "coco", "train")
    val_dir = os.path.join(base_path, "coco", "valid")

    # Define annotation files
    train_annotations_file = os.path.join(train_dir, "_annotations.coco.json")
    val_annotations_file = os.path.join(val_dir, "_annotations.coco.json")

    # Create directories if they don't exist
    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"Data path: {base_path}")
    print(f"Train path: {train_dir}")
    print(f"Validation path: {val_dir}")

    # Check if annotation files exist
    missing_files = []
    for file_path, name in [(train_annotations_file, "Training"),
                            (val_annotations_file, "Validation")]:
        if not os.path.exists(file_path):
            missing_files.append((name, file_path))

    if missing_files:
        for name, path in missing_files:
            print(f"Warning: {name} annotations file not found at {path}")
        print("Please ensure all annotation files exist before proceeding.")

    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(models_path, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Create log file
    log_file = os.path.join(checkpoint_dir, "training_log.txt")

    # Define log function
    def log_message(message):
        """Write message to log file and print to console"""
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

    # Log initial message
    log_message(f"=== Training started at {timestamp} ===")

    return train_dir, val_dir, train_annotations_file, val_annotations_file, checkpoint_dir, log_message, timestamp


def load_datasets(train_dir, val_dir, train_annotations_file, val_annotations_file, log_message):
    """
    Load datasets and create data loaders

    Args:
        train_dir (str): Directory containing training data
        val_dir (str): Directory containing validation data
        train_annotations_file (str): Path to training annotations file
        val_annotations_file (str): Path to validation annotations file
        log_message (function): Function to log messages

    Returns:
        tuple: Contains the following elements:
            - train_loader (DataLoader): DataLoader for training data
            - val_loader (DataLoader or None): DataLoader for validation data
            - num_classes (int): Number of classes in the dataset (including background)
    """
    log_message("Loading datasets...")

    # Training dataset
    train_dataset = LeafDataset(train_dir, train_annotations_file, get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=14,  # Increased for 128GB RAM
        shuffle=True,
        num_workers=8,  # Increased workers for better I/O throughput
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4  # Increased prefetching
    )
    log_message(f"Training dataset loaded with {len(train_dataset)} images")

    # Get number of classes from training dataset
    num_classes = len(train_dataset.coco.coco.getCatIds()) + 1  # +1 for background
    log_message(f"Number of classes: {num_classes}")

    # Validation dataset
    val_dataset = None
    val_loader = None
    if os.path.exists(val_annotations_file):
        val_dataset = LeafDataset(val_dir, val_annotations_file, get_transform(train=False))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=18,  # Increased for 128GB RAM
            shuffle=False,
            num_workers=6,  # Increased workers
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
        log_message(f"Validation dataset loaded with {len(val_dataset)} images")
    else:
        log_message("WARNING: Validation dataset not available")

    # Check if the training dataset is not empty
    if len(train_dataset) == 0:
        log_message("ERROR: Training dataset is empty!")
        raise ValueError("Training dataset contains no images")

    return train_loader, val_loader, num_classes

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

def setup_model_and_optimization(num_classes, device, log_message, base_checkpoint_dir=None):
    """
    Initialize model, optimizer, scheduler, and load checkpoint if available

    Args:
        num_classes (int): Number of classes for model output
        device (torch.device): Device to run model on (cuda, mps, or cpu)
        log_message (function): Function to log messages
        base_checkpoint_dir (str, optional): Directory to look for checkpoints. If None, starts from scratch.

    Returns:
        tuple: Contains the following elements:
            - model (nn.Module): Initialized model (moved to device)
            - optimizer (torch.optim.Optimizer): Model optimizer
            - lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            - start_epoch (int): Epoch to start training from (0 if fresh start)
            - best_val_loss (float): Best validation loss from previous training
            - early_stopping_counter (int): Counter for early stopping logic
    """
    # Initialize model
    log_message("Initializing model...")
    model = get_instance_segmentation_model(num_classes)
    model.to(device)
    log_message(f"Model initialized and moved to device: {device}")

    # Set default values
    start_epoch = 0
    best_val_loss = float('inf')
    early_stopping_counter = 0
    checkpoint = None

    # Check for existing checkpoints
    if base_checkpoint_dir is not None and os.path.exists(base_checkpoint_dir):
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
                    early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                    log_message(f"Resuming from epoch {start_epoch} with early stopping counter {early_stopping_counter}")
                except Exception as e:
                    log_message(f"Error loading checkpoint: {e}")
                    log_message("Starting training from scratch")
                    start_epoch = 0

    # Set up optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 
        lr=0.004,  # Adjusted for larger batch size
        momentum=0.9,
        weight_decay=0.0004  # Adjusted for ResNet101
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=3,  # Keep at 3 since we can process more data per epoch
        gamma=0.1
    )

    # Load optimizer and scheduler states if resuming
    if checkpoint is not None and start_epoch > 0:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_message("Loaded optimizer and scheduler states")
        except Exception as e:
            log_message(f"Error loading optimizer/scheduler states: {e}")

    log_message("Optimizer and learning rate scheduler initialized")

    return model, optimizer, lr_scheduler, start_epoch, best_val_loss, early_stopping_counter


def train_epoch(model, optimizer, train_loader, device, log_message, scaler=None, epoch_num=None, total_epochs=None):
    """

    Run a single training epoch

    Args:
        model (nn.Module): The model to train
        optimizer (torch.optim.Optimizer): The optimizer to use
        train_loader (DataLoader): DataLoader for training data
        device (torch.device): Device to run model on
        log_message (function): Function to log messages
        epoch_num (int, optional): Current epoch number for logging
        total_epochs (int, optional): Total number of epochs for logging

    Returns:
        float: Average training loss for the epoch
    """
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    batch_count = 0

    if epoch_num is not None and total_epochs is not None:
        log_message(f"Epoch {epoch_num}/{total_epochs}")

    for i, (images, targets) in enumerate(train_loader):
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients
        optimizer.zero_grad()

        # Use mixed precision if on CUDA and scaler is provided
        if device.type == 'cuda' and scaler is not None:
            # Forward pass with autocast
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            # Backward pass with scaling
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular precision training
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()

        epoch_loss += losses.item()
        batch_count += 1

        if (i + 1) % 10 == 0:
            log_message(f"  Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")

        # Clean up memory
        del images, targets, loss_dict, losses
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0

    if epoch_num is not None:
        log_message(f"  Epoch {epoch_num} training completed. Average Loss: {avg_train_loss:.4f}")
        epoch_duration = time.time() - epoch_start_time
        log_message(f"  Epoch duration: {epoch_duration:.2f} seconds")

    return avg_train_loss

def early_stopping_check(val_loss, best_val_loss, counter, patience, min_delta, log_message):
    """
    Check early stopping criteria and update counter

    Args:
        val_loss (float or None): Current validation loss (None if validation not performed)
        best_val_loss (float): Best validation loss seen so far
        counter (int): Current early stopping counter
        patience (int): Maximum number of epochs to wait for improvement
        min_delta (float): Minimum change to qualify as improvement
        log_message (function): Function to log messages

    Returns:
        tuple: Contains the following elements:
            - best_val_loss (float): Updated best validation loss
            - counter (int): Updated early stopping counter
            - should_stop (bool): Whether training should stop
            - improved (bool): Whether validation loss improved
    """
    should_stop = False
    improved = False

    # Skip early stopping if no validation loss is available
    if val_loss is None:
        log_message("  Early stopping skipped: No validation loss available")
        return best_val_loss, counter, should_stop, improved

    # Check if we have significant improvement
    if val_loss < best_val_loss - min_delta:
        # Improvement found, reset counter
        improved = True
        best_val_loss = val_loss
        counter = 0
        log_message(f"  Validation loss improved to {val_loss:.4f}. Early stopping counter reset.")
    else:
        # No significant improvement
        counter += 1
        log_message(f"  No significant improvement. Early stopping counter: {counter}/{patience}")

    # Check if we should stop training
    if counter >= patience:
        should_stop = True
        log_message(f"  Early stopping triggered after {counter} epochs without improvement")

    return best_val_loss, counter, should_stop, improved


def save_checkpoint(model, optimizer, scheduler, checkpoint_dir, epoch, losses, counters, metadata=None, is_best=False):
    """
    Save model checkpoint

    Args:
        model (nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to save
        checkpoint_dir (str): Directory to save checkpoint in
        epoch (int): Current epoch number
        losses (dict): Dictionary containing loss values (e.g., {'train': 0.1, 'val': 0.2})
        counters (dict): Dictionary containing counters (e.g., {'early_stopping': 2, 'num_classes': 5})
        metadata (dict, optional): Additional metadata to save
        is_best (bool, optional): Whether this is the best model so far

    Returns:
        str: Path to the saved checkpoint file
    """
    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    # Add losses
    if losses:
        checkpoint.update(losses)

    # Add counters
    if counters:
        checkpoint.update(counters)

    # Add additional metadata
    if metadata:
        checkpoint.update(metadata)

    # Regular checkpoint filename
    checkpoint_filename = f'mask_rcnn_checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)

    # If this is the best model, also save it as the best model
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'mask_rcnn_best_model.pth')
        torch.save(checkpoint, best_model_path)
        return best_model_path

    return checkpoint_path


def main():
    # Set the path to your annotated images and annotations
    data_path = "/workdir/data/grape/grape_pheno/grape_peduncle/data/annotations"
    checkpoint_path = "/workdir/data/grape/grape_pheno/grape_peduncle/checkpoints/maskrcnn/run_5.1"

    # Setup paths and logging
    train_dir, val_dir, train_annotations_file, val_annotations_file, checkpoint_dir, log_message, timestamp = setup_paths_and_logging(data_path, checkpoint_path)

    # Load datasets
    train_loader, val_loader, num_classes = load_datasets(
        train_dir, val_dir, train_annotations_file, val_annotations_file, log_message
    )

    # Set up device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    log_message(f"Using device: {device}")

    # Set up model and optimization
    base_checkpoint_dir = os.path.join(checkpoint_path, "checkpoints")
    model, optimizer, lr_scheduler, start_epoch, best_val_loss, early_stopping_counter = setup_model_and_optimization(
        num_classes, device, log_message, base_checkpoint_dir
    )

    # Set up GradScaler for mixed precision training (for CUDA only)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    # Initiate early stopping parameters
    early_stopping_patience = 4  # Increased from 3 to 4
    early_stopping_min_delta = 0.0005  # Reduced from 0.001 for finer improvement detection

    # Training parameters
    num_epochs = 8  # Increased from 5 to 8 for better convergence
    log_message(f"Starting training for {num_epochs} epochs")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        # Run training epoch
        avg_train_loss = train_epoch(
            model,
            optimizer,
            train_loader,
            device,
            log_message,
            scaler=scaler,
            epoch_num=epoch+1,
            total_epochs=num_epochs
        )

        # Validation phase
        val_loss = None
        if val_loader:
            val_loss = evaluate_model(model, val_loader, device, log_message)
            log_message(f"  Validation Loss: {val_loss:.4f}")
        else:
            log_message("WARNING: Validation dataset not available")

        # Update learning rate
        lr_scheduler.step()
        log_message(f"  Learning rate updated to: {optimizer.param_groups[0]['lr']:.6f}")

        # Check early stopping criteria
        best_val_loss, early_stopping_counter, should_stop, improved = early_stopping_check(
            val_loss,
            best_val_loss,
            early_stopping_counter,
            early_stopping_patience,
            early_stopping_min_delta,
            log_message
        )

        # Prepare losses and counters for checkpoint
        losses = {
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }

        counters = {
            'early_stopping_counter': early_stopping_counter,
            'num_classes': num_classes
        }

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            checkpoint_dir,
            epoch + 1,
            losses,
            counters,
            is_best=improved
        )
        log_message(f"  Checkpoint saved to {checkpoint_path}")

        # Break training loop if early stopping triggered
        if should_stop:
            log_message("Early stopping triggered. Halting training.")
            break

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "maskrcnn_model_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'train_loss': avg_train_loss,
        'val_loss': val_loss if val_loader else None
    }, final_model_path)
    log_message(f"Final model saved to {final_model_path}")

    # Create symlink to latest checkpoint directory
    create_symlink(checkpoint_dir, log_message)

    # Clean up
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()

    # Final message
    log_message("Training completed!")

if __name__ == "__main__":
    main()
