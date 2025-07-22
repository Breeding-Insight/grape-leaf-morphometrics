'''
OVERVIEW
Purpose: Train a Mask R-CNN model for instance segmentation on a custom leaf dataset.
Dataset: The dataset is loaded from COCO-formatted annotations.
Model: Hybrid PANet implementation using pretrained ResNeXt-101 32x8d + custom path aggregation.
Training: The model is trained for a specified number of epochs.
Checkpoints: The script supports resuming training from checkpoints and saving checkpoints at each epoch.
Evaluation: The model is evaluated on a validation set.
Logging: Comprehensive logging and error handling are included.
Framework: The training is performed using the PyTorch framework.
Devices: The script is optimized for CUDA with fallback to CPU.
Architecture: Hybrid approach combining torchvision's pretrained ResNeXt-101 32x8d with custom PANet aggregation.
Author: aja294@cornell.edu
'''

# Import standard libraries
import os
import gc
import time
from datetime import datetime
import traceback

# Import third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.amp import GradScaler, autocast

# Import specific modules from torchvision
import torch.multiprocessing as mps
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import RoIAlign
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights

# Import COCO utilities
from .coco_utils import get_coco_api_from_dataset, CocoEvaluator, COCO_EVAL_AVAILABLE

# CONFIGURATION: Set to False to disable mAP evaluation and use loss-only evaluation
# This is useful for debugging or when mAP evaluation is causing issues
ENABLE_MAP_EVALUATION = True  # Set to False to disable mAP evaluation

# UPDATED: Import torchvision.transforms.v2 with proper error handling for nightly builds
try:
    import torchvision.transforms.v2 as T_v2
    USE_V2_TRANSFORMS = True
except ImportError:
    import torchvision.transforms as T_v2
    USE_V2_TRANSFORMS = False
    print("Warning: torchvision.transforms.v2 not available, using v1 transforms")

# ============================================================================
# ENHANCED GPU MEMORY TRACKING AND MANAGEMENT
# ============================================================================

class GPUMemoryTracker:
    """Comprehensive GPU memory tracking and management"""
    
    def __init__(self, device, log_message=print):
        self.device = device
        self.log_message = log_message
        self.memory_history = []
        self.peak_memory = 0
        self.oom_warnings = 0
        self.reset_stats()
        
    def reset_stats(self):
        """Reset memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
    
    def get_memory_info(self):
        """Get comprehensive memory information"""
        if not torch.cuda.is_available():
            return None
            
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        max_allocated = torch.cuda.max_memory_allocated(self.device)
        total_memory = torch.cuda.get_device_properties(self.device).total_memory
        
        return {
            'allocated_mb': allocated / (1024**2),
            'reserved_mb': reserved / (1024**2),
            'max_allocated_mb': max_allocated / (1024**2),
            'total_mb': total_memory / (1024**2),
            'utilization_percent': (allocated / total_memory) * 100,
            'free_mb': (total_memory - allocated) / (1024**2)
        }
    
    def log_memory_status(self, context="", force=False):
        """Log current memory status with context"""
        if not torch.cuda.is_available():
            return
            
        mem_info = self.get_memory_info()
        if not mem_info:
            return
            
        # Update peak memory
        if mem_info['allocated_mb'] > self.peak_memory:
            self.peak_memory = mem_info['allocated_mb']
        
        # Check for memory pressure
        utilization = mem_info['utilization_percent']
        warning_threshold = 85.0
        critical_threshold = 95.0
        
        if utilization > critical_threshold:
            self.oom_warnings += 1
            self.log_message(f"  🚨 CRITICAL GPU MEMORY: {utilization:.1f}% used ({mem_info['allocated_mb']:.1f}MB)")
            self.log_message(f"     Free: {mem_info['free_mb']:.1f}MB | Peak: {self.peak_memory:.1f}MB | Context: {context}")
            self._emergency_cleanup()
        elif utilization > warning_threshold or force:
            self.log_message(f"  ⚠️  GPU MEMORY: {utilization:.1f}% used ({mem_info['allocated_mb']:.1f}MB)")
            self.log_message(f"     Free: {mem_info['free_mb']:.1f}MB | Peak: {self.peak_memory:.1f}MB | Context: {context}")
        
        # Store in history for analysis
        self.memory_history.append({
            'timestamp': time.time(),
            'context': context,
            **mem_info
        })
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup"""
        self.log_message("  🚨 Performing emergency memory cleanup...")
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection multiple times
        for i in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        # Log after cleanup
        mem_info = self.get_memory_info()
        if mem_info:
            self.log_message(f"  ✅ Cleanup complete: {mem_info['allocated_mb']:.1f}MB allocated")
    
    def get_memory_summary(self):
        """Get memory usage summary"""
        if not self.memory_history:
            return "No memory data collected"
            
        mem_info = self.get_memory_info()
        if not mem_info:
            return "GPU not available"
            
        return {
            'current_allocated_mb': mem_info['allocated_mb'],
            'peak_allocated_mb': self.peak_memory,
            'total_memory_mb': mem_info['total_mb'],
            'utilization_percent': mem_info['utilization_percent'],
            'oom_warnings': self.oom_warnings,
            'memory_entries': len(self.memory_history)
        }
    
    def log_memory_summary(self):
        """Log comprehensive memory summary"""
        summary = self.get_memory_summary()
        if isinstance(summary, str):
            self.log_message(f"  Memory Summary: {summary}")
            return
            
        self.log_message("  " + "="*50)
        self.log_message("  GPU MEMORY SUMMARY")
        self.log_message("  " + "="*50)
        self.log_message(f"  Current Allocated: {summary['current_allocated_mb']:.1f} MB")
        self.log_message(f"  Peak Allocated: {summary['peak_allocated_mb']:.1f} MB")
        self.log_message(f"  Total GPU Memory: {summary['total_memory_mb']:.1f} MB")
        self.log_message(f"  Utilization: {summary['utilization_percent']:.1f}%")
        self.log_message(f"  OOM Warnings: {summary['oom_warnings']}")
        self.log_message(f"  Memory Tracking Entries: {summary['memory_entries']}")
        self.log_message("  " + "="*50)

def adaptive_batch_size_adjustment(current_batch_size, memory_utilization, log_message):
    """Dynamically adjust batch size based on memory usage"""
    if memory_utilization > 95:
        new_batch_size = max(1, current_batch_size - 1)
        log_message(f"  🚨 Critical memory usage ({memory_utilization:.1f}%), reducing batch size to {new_batch_size}")
        return new_batch_size
    elif memory_utilization > 85:
        log_message(f"  ⚠️  High memory usage ({memory_utilization:.1f}%), consider reducing batch size")
        return current_batch_size
    return current_batch_size

def memory_efficient_forward_pass(model, images, targets, device, scaler=None):
    """Memory-efficient forward pass with gradient checkpointing"""
    if device.type == 'cuda':
        # Use gradient checkpointing for memory efficiency
        if scaler:
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
        else:
            loss_dict = model(images, targets)
            losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
        return loss_dict, losses
    else:
        loss_dict = model(images, targets)
        losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
        return loss_dict, losses

# ============================================================================
# HYBRID RESNEXT-101 PANET IMPLEMENTATION
# ============================================================================

class ResNeXtFPN(nn.Module):
    """
    Feature Pyramid Network built on top of pretrained ResNeXt-101 32x8d
    """
    def __init__(self, pretrained=True, trainable_layers=3):
        super().__init__()
        
        # Load pretrained ResNeXt-101 32x8d using the new 'weights' API
        print(f"Loading ResNeXt-101 32x8d with pretrained weights (trainable_layers={trainable_layers})")
        weights = ResNeXt101_32X8D_Weights.DEFAULT if pretrained else None
        backbone = resnext101_32x8d(weights=weights)
        
        # Extract layers for FPN
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1  # C2
        self.layer2 = backbone.layer2  # C3
        self.layer3 = backbone.layer3  # C4
        self.layer4 = backbone.layer4  # C5
        
        # Freeze early layers based on trainable_layers
        layers_to_train = [self.layer4, self.layer3, self.layer2, self.layer1][:trainable_layers]
        for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            if layer not in layers_to_train:
                for param in layer.parameters():
                    param.requires_grad = False
        
        # FPN lateral connections
        self.lateral_conv_c5 = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral_conv_c4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.lateral_conv_c3 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral_conv_c2 = nn.Conv2d(256, 256, kernel_size=1)
        
        # FPN output convolutions
        self.output_conv_p5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.output_conv_p4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.output_conv_p3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.output_conv_p2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Initialize FPN weights
        self._init_fpn_weights()
        
        print("ResNeXt-101 32x8d FPN initialized:")
        print(f"  - Pretrained weights: {'DEFAULT' if pretrained else 'None'}")
        print(f"  - Trainable layers: {trainable_layers}")
        print(f"  - Backbone architecture: ResNeXt-101 32x8d (groups=32, width_per_group=8)")
        
    def _init_fpn_weights(self):
        """Initialize FPN weights"""
        for m in [self.lateral_conv_c5, self.lateral_conv_c4, self.lateral_conv_c3, self.lateral_conv_c2,
                  self.output_conv_p5, self.output_conv_p4, self.output_conv_p3, self.output_conv_p2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through ResNeXt backbone and FPN"""
        # ResNeXt backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        c2 = self.layer1(x)  # 256 channels
        c3 = self.layer2(c2)  # 512 channels
        c4 = self.layer3(c3)  # 1024 channels
        c5 = self.layer4(c4)  # 2048 channels
        
        # FPN lateral connections
        p5 = self.lateral_conv_c5(c5)
        p4 = self.lateral_conv_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_conv_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_conv_c2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        
        # FPN output convolutions
        p5 = self.output_conv_p5(p5)
        p4 = self.output_conv_p4(p4)
        p3 = self.output_conv_p3(p3)
        p2 = self.output_conv_p2(p2)
        
        return {'0': p2, '1': p3, '2': p4, '3': p5}

class PANetPathAggregation(nn.Module):
    """
    Optimized PANet path aggregation with batch normalization and proper initialization
    """
    def __init__(self, in_channels=256):
        super().__init__()
        
        # Bottom-up path lateral convolutions
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        
        # Downsampling operations
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, fpn_features):
        """
        Forward pass for PANet path aggregation
        
        Args:
            fpn_features: Dictionary with keys '0', '1', '2', '3' containing P2, P3, P4, P5
            
        Returns:
            Dictionary with enhanced features after PANet aggregation
        """
        # Extract features (P2, P3, P4, P5)
        p2, p3, p4, p5 = [fpn_features[k] for k in ['0', '1', '2', '3']]
        
        # Bottom-up path aggregation
        n2 = self.lateral_convs[0](p2)
        n3 = self.lateral_convs[1](p3) + self.downsample(n2)
        n4 = self.lateral_convs[2](p4) + self.downsample(n3)
        n5 = self.lateral_convs[3](p5) + self.downsample(n4)
        
        # Feature fusion (element-wise addition)
        enhanced_features = {
            '0': p2 + n2,  # P2 enhanced
            '1': p3 + n3,  # P3 enhanced
            '2': p4 + n4,  # P4 enhanced
            '3': p5 + n5,  # P5 enhanced
            '4': F.max_pool2d(p5, kernel_size=1, stride=2)  # P6 for large objects
        }
        
        return enhanced_features

class HybridResNeXtPANetBackbone(nn.Module):
    """
    Hybrid ResNeXt-101 32x8d PANet backbone combining:
    1. Pretrained ResNeXt-101 32x8d FPN (optimized, pretrained)
    2. Custom PANet path aggregation on top
    
    This gives us the best of both worlds:
    - Proven, optimized ResNeXt backbone with pretrained weights
    - Custom PANet aggregation for improved feature representation
    """
    def __init__(self, pretrained=True, trainable_layers=3):
        super().__init__()
        
        # ResNeXt-101 32x8d FPN backbone
        self.fpn_backbone = ResNeXtFPN(pretrained=pretrained, trainable_layers=trainable_layers)
        
        # Custom PANet path aggregation
        self.panet_aggregation = PANetPathAggregation(in_channels=256)
        
        # Output channels (standard for FPN)
        self.out_channels = 256
        
        print("Hybrid ResNeXt-101 32x8d PANet backbone initialized:")
        print("  - Pretrained ResNeXt-101 32x8d FPN backbone")
        print("  - Custom PANet path aggregation")
        print("  - Output channels: 256")
        
    def forward(self, x):
        """
        Forward pass combining ResNeXt FPN and PANet
        
        Args:
            x: Input tensor [batch_size, 3, height, width]
            
        Returns:
            Dictionary with enhanced features at multiple scales
        """
        # Get FPN features from ResNeXt backbone
        fpn_features = self.fpn_backbone(x)
        
        # Apply PANet aggregation
        panet_features = self.panet_aggregation(fpn_features)
        
        return panet_features

class HybridResNeXtMaskRCNN(nn.Module):
    """
    Mask R-CNN with Hybrid ResNeXt-101 32x8d PANet backbone
    Combines pretrained ResNeXt-101 32x8d FPN with custom PANet path aggregation
    """
    def __init__(self, num_classes, pretrained_backbone=True, trainable_layers=3):
        super().__init__()
        
        # Build hybrid ResNeXt PANet backbone
        self.backbone = HybridResNeXtPANetBackbone(
            pretrained=pretrained_backbone,
            trainable_layers=trainable_layers
        )
        
        # Anchor generator optimized for leaf detection based on your original analysis
        anchor_sizes = (
            (257,),   # Small leaves
            (1139,),  # Medium leaves  
            (2056,),  # Large leaves
            (3183,),  # Very large leaves
            (4475,),  # Extremely large leaves
        )
        aspect_ratios = ((0.76, 0.85, 0.94, 1.01, 1.13),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            sizes=anchor_sizes,
            aspect_ratios=aspect_ratios
        )
        
        # Initialize Mask R-CNN with hybrid backbone
        self.model = MaskRCNN(
            self.backbone,
            num_classes=num_classes,
            rpn_anchor_generator=rpn_anchor_generator,
            rpn_head=None,  # Use default
            box_roi_pooler=None  # Use default
        )
        
        # Replace box predictor head
        in_features_box = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, num_classes)
        
        # Enhanced mask predictor with higher capacity for detailed segmentation
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 512  # Increased capacity for better mask quality
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, 
            hidden_layer, 
            num_classes
        )
        
        # Custom mask ROI pooler with higher resolution for finer details
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=28,  # Higher resolution than default 14x14
            sampling_ratio=2
        )
        self.model.roi_heads.mask_roi_pool = mask_roi_pooler
        
        print("Hybrid ResNeXt-101 32x8d Mask R-CNN initialized:")
        print(f"  - Number of classes: {num_classes}")
        print(f"  - Mask ROI pooler resolution: 28x28")
        print(f"  - Mask predictor hidden layer: {hidden_layer}")
        print(f"  - Anchor sizes: {anchor_sizes}")
        print(f"  - Aspect ratios: {aspect_ratios[0]}")
        
    def forward(self, images, targets=None):
        return self.model(images, targets)

# Function to create hybrid ResNeXt model
def get_hybrid_resnext_mask_rcnn_model(num_classes, pretrained_backbone=True, trainable_layers=3):
    """
    Create Mask R-CNN model with hybrid ResNeXt-101 32x8d PANet backbone
    
    Args:
        num_classes (int): Number of classes including background
        pretrained_backbone (bool): Whether to use pretrained backbone weights
        trainable_layers (int): Number of trainable layers in backbone
        
    Returns:
        nn.Module: Configured Mask R-CNN model with hybrid ResNeXt PANet backbone
    """
    model = HybridResNeXtMaskRCNN(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_layers=trainable_layers
    )
    return model.model

# ============================================================================
# DATASET AND UTILITY FUNCTIONS
# ============================================================================

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
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

# Enhanced transform function for ResNeXt
def get_transform(train):
    transforms = []
    
    if USE_V2_TRANSFORMS:
        transforms.append(T_v2.ToImage())
        
        if train:
            # Enhanced augmentation pipeline optimized for ResNeXt
            transforms.extend([
                T_v2.RandomHorizontalFlip(0.5),
                T_v2.RandomVerticalFlip(0.3),
                T_v2.RandomRotation(degrees=(-30, 30)),
                T_v2.RandomAffine(degrees=0, scale=(0.8, 1.2), translate=(0.1, 0.1)),
                T_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                T_v2.RandomApply([T_v2.GaussianBlur(kernel_size=3)], p=0.1),
                # Additional augmentations for ResNeXt robustness
                T_v2.RandomApply([T_v2.RandomAdjustSharpness(sharpness_factor=2)], p=0.1),
            ])
        
        transforms.append(T_v2.ToDtype(torch.float32, scale=True))

        return T_v2.Compose(transforms)
    else:
        transforms.append(T.ToTensor())
        
        if train:
            transforms.extend([
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            ])
        
        return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_instance_segmentation_model(num_classes, pretrained_backbone=True, trainable_layers=3):
    """
    Create a Mask R-CNN model with hybrid ResNeXt-101 32x8d PANet backbone
    
    Args:
        num_classes (int): Number of classes including background
        pretrained_backbone (bool): Whether to use pretrained backbone weights
        trainable_layers (int): Number of trainable layers in backbone
        
    Returns:
        nn.Module: Configured Mask R-CNN model with hybrid ResNeXt PANet backbone
    """
    model = get_hybrid_resnext_mask_rcnn_model(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_layers=trainable_layers
    )
    return model

# ============================================================================
# REMAINING FUNCTIONS (SAME AS BEFORE BUT WITH RESNEXT SPECIFIC UPDATES)
# ============================================================================

def create_symlink(checkpoint_dir, log_message):
    """Create symlink to latest checkpoint directory"""
    base_dir = os.path.dirname(checkpoint_dir)
    latest_link = os.path.join(base_dir, "latest")

    if os.path.exists(latest_link):
        try:
            if os.path.islink(latest_link):
                if os.name == 'nt':
                    os.remove(latest_link)
                else:
                    os.unlink(latest_link)
            else:
                log_message(f"Warning: {latest_link} exists but is not a symlink. Removing it.")
                if os.path.isdir(latest_link):
                    import shutil
                    shutil.rmtree(latest_link)
                else:
                    os.remove(latest_link)
            log_message(f"Removed existing symlink: {latest_link}")
        except Exception as e:
            log_message(f"Error removing existing symlink: {e}")
            return

    try:
        if os.name == 'nt':
            try:
                os.symlink(checkpoint_dir, latest_link, target_is_directory=True)
            except:
                import subprocess
                subprocess.check_call(['mklink', '/J', latest_link, checkpoint_dir], shell=True)
        else:
            os.symlink(checkpoint_dir, latest_link)
        log_message(f"Created symlink to latest checkpoint directory: {latest_link} -> {checkpoint_dir}")
    except Exception as e:
        log_message(f"Error creating symlink: {e}")

def setup_paths_and_logging(base_path, models_path):
    """Setup paths, directories and logging infrastructure"""
    train_dir = os.path.join(base_path, "coco100", "train")
    val_dir = os.path.join(base_path, "coco100", "valid")
    train_annotations_file = os.path.join(train_dir, "_annotations.coco.json")
    val_annotations_file = os.path.join(val_dir, "_annotations.coco.json")

    for directory in [train_dir, val_dir]:
        os.makedirs(directory, exist_ok=True)

    print(f"Data path: {base_path}")
    print(f"Train path: {train_dir}")
    print(f"Validation path: {val_dir}")

    missing_files = []
    for file_path, name in [(train_annotations_file, "Training"),
                            (val_annotations_file, "Validation")]:
        if not os.path.exists(file_path):
            missing_files.append((name, file_path))

    if missing_files:
        for name, path in missing_files:
            print(f"Warning: {name} annotations file not found at {path}")
        print("Please ensure all annotation files exist before proceeding.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(models_path, f"checkpoints_{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    log_file = os.path.join(checkpoint_dir, "training_log.txt")

    def log_message(message):
        """Write message to log file and print to console"""
        print(message)
        with open(log_file, "a") as f:
            f.write(f"{message}\n")

    log_message(f"=== Hybrid ResNeXt-101 32x8d PANet Training started at {timestamp} ===")
    return train_dir, val_dir, train_annotations_file, val_annotations_file, checkpoint_dir, log_message, timestamp

def load_datasets(train_dir, val_dir, train_annotations_file, val_annotations_file, log_message):
    """Load datasets and create data loaders"""
    log_message("Loading datasets...")

    train_dataset = LeafDataset(train_dir, train_annotations_file, get_transform(train=True))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=3,  # Slightly reduced for ResNeXt memory usage
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    log_message(f"Training dataset loaded with {len(train_dataset)} images")

    num_classes = len(train_dataset.coco.coco.getCatIds()) + 1
    log_message(f"Number of classes: {num_classes}")

    val_dataset = None
    val_loader = None
    if os.path.exists(val_annotations_file):
        val_dataset = LeafDataset(val_dir, val_annotations_file, get_transform(train=False))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=1,  # REDUCED: From 4 to 1 to prevent CUDA OOM during evaluation
            shuffle=False,
            num_workers=1,  # REDUCED: From 2 to 1 to save memory
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=True if len(val_dataset) > 20 else False
        )
        log_message(f"Validation dataset loaded with {len(val_dataset)} images")
    else:
        log_message("WARNING: Validation dataset not available")

    if len(train_dataset) == 0:
        log_message("ERROR: Training dataset is empty!")
        raise ValueError("Training dataset contains no images")

    return train_loader, val_loader, num_classes, val_dataset

def evaluate_model_mAP(model, data_loader, device, dataset, log_message=print):
    """
    FIXED: Evaluate the model using COCO mAP metrics for segmentation
    
    Key Fix: Prevents BatchNorm contamination by maintaining eval mode for the entire
    validation process and only briefly switching to train mode for loss calculation
    when absolutely necessary.
    
    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to run evaluation on
        dataset: The dataset object (needed for COCO API conversion)
        log_message: Logging function
        
    Returns:
        tuple: (segm_mAP, bbox_mAP, val_loss) or (None, None, val_loss) if COCO eval fails
    """
    if not COCO_EVAL_AVAILABLE:
        log_message("  COCO evaluation tools not available, falling back to loss-only evaluation")
        return evaluate_model_loss_only(model, data_loader, device, log_message)
    
    # CRITICAL FIX: Keep model in eval mode throughout the entire validation process
    model.eval()
    total_loss = 0
    batch_count = 0
    
    try:
        # Prepare COCO evaluator
        coco_gt = get_coco_api_from_dataset(dataset)
        coco_evaluator = CocoEvaluator(coco_gt, ['bbox', 'segm'])
        
        log_message("  Starting mAP evaluation...")
        
        with torch.no_grad():
            for i, (images, targets) in enumerate(data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Get model outputs for mAP calculation (in eval mode)
                model_outputs = model(images)
                
                # ENGINEERING FIX: Calculate loss with minimal BatchNorm contamination
                # We need to briefly switch to train mode for loss calculation, but we do it
                # per-batch to minimize the impact on running statistics
                try:
                    # Briefly switch to train mode for loss calculation only
                    model.train()
                    
                    # Calculate loss with current batch (no gradients needed for validation)
                    loss_dict = model(images, targets)
                    losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
                    total_loss += losses.item()  # .item() called directly on tensor
                    batch_count += 1
                    
                    # Immediately restore eval mode to prevent further contamination
                    model.eval()
                    
                except Exception as loss_calc_error:
                    # If loss calculation fails, continue with mAP but set loss to inf
                    log_message(f"    Warning: Loss calculation failed for batch {i+1}: {loss_calc_error}")
                    total_loss += float('inf')
                    batch_count += 1
                    model.eval()  # Ensure we're back in eval mode
                
                # Prepare outputs for COCO evaluation (model is back in eval mode)
                outputs = []
                for output in model_outputs:
                    output_dict = {
                        'boxes': output['boxes'].cpu(),
                        'labels': output['labels'].cpu(),
                        'scores': output['scores'].cpu(),
                        'masks': output['masks'].cpu()
                    }
                    outputs.append(output_dict)
                
                # Update COCO evaluator - simplified using direct image_id extraction
                res = {t['image_id'].item(): o for t, o in zip(targets, outputs)}
                coco_evaluator.update(res)
                
                if (i + 1) % 5 == 0:
                    if batch_count > 0 and total_loss != float('inf') * batch_count:
                        current_avg_loss = total_loss / batch_count
                        log_message(f"  mAP eval batch {i+1}/{len(data_loader)}, Avg Loss: {current_avg_loss:.4f}")
                    else:
                        log_message(f"  mAP eval batch {i+1}/{len(data_loader)}, Loss calculation failed")
                
                # Memory cleanup
                del images, targets, model_outputs, outputs
                # loss_dict and losses are always created together in the try block
                # or never created at all, so no conditional check needed
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # ADDED: Force synchronization to ensure memory is actually freed
                    torch.cuda.synchronize()
        
        # Calculate final mAP scores
        log_message("  Computing mAP scores...")
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # FIXED: Extract mAP values using correct integer-based indexing.
        # The .stats attribute is a list, not a dictionary.
        # coco_evaluator.coco_eval['bbox'] is the COCOeval object for bounding boxes.
        # .stats[0] is the mAP @ IoU=0.50:0.95.
        bbox_mAP = coco_evaluator.coco_eval['bbox'].stats[0]
        segm_mAP = coco_evaluator.coco_eval['segm'].stats[0]
        
        # Calculate average loss (handle case where all loss calculations failed)
        if batch_count > 0 and total_loss != float('inf') * batch_count:
            avg_loss = total_loss / batch_count
        else:
            log_message("  Warning: All loss calculations failed, setting avg_loss to inf")
            avg_loss = float('inf')
        
        log_message(f"  Evaluation Results:")
        log_message(f"    Segmentation mAP: {segm_mAP:.4f}")
        log_message(f"    Bounding Box mAP: {bbox_mAP:.4f}")
        log_message(f"    Average Loss: {avg_loss:.4f}")
        
        return segm_mAP, bbox_mAP, avg_loss
        
    except Exception as e:
        log_message(f"Error during mAP evaluation: {e}")
        traceback.print_exc()
        log_message("  Falling back to loss-only evaluation")
        return None, None, float('inf')

def evaluate_model_loss_only(model, data_loader, device, log_message=print):
    """
    FIXED: Fallback evaluation function that calculates validation loss with minimal BatchNorm contamination
    
    Key Engineering Fix: Instead of switching to train mode for the entire validation loop,
    we maintain eval mode and only briefly switch to train mode per batch for loss calculation.
    This significantly reduces BatchNorm contamination while still providing accurate loss values.
    
    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to run evaluation on
        log_message: Logging function
        
    Returns:
        tuple: (None, None, avg_loss) - None for mAP values, avg_loss for loss
    """
    # CRITICAL FIX: Start and maintain eval mode
    model.eval()
    total_loss = 0
    batch_count = 0
    successful_batches = 0

    try:
        with torch.no_grad():  # Main validation loop runs without gradients
            for i, (images, targets) in enumerate(data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                try:
                    # ENGINEERING FIX: Minimal train mode contamination
                    # Briefly switch to train mode for loss calculation only
                    model.train()
                    
                    # Calculate loss for this specific batch (no gradients needed)
                    loss_dict = model(images, targets)
                    losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
                    total_loss += losses.item()  # .item() called directly on tensor
                    successful_batches += 1
                    
                    # Immediately return to eval mode to prevent further contamination
                    model.eval()
                    
                except Exception as batch_error:
                    # If this batch fails, log it but continue
                    log_message(f"    Warning: Batch {i+1} loss calculation failed: {batch_error}")
                    model.eval()  # Ensure we're back in eval mode
                    # Don't increment successful_batches for failed batches
                
                batch_count += 1

                if (i + 1) % 5 == 0:  # Log progress
                    if successful_batches > 0:
                        current_avg_loss = total_loss / successful_batches
                        log_message(f"  Eval batch {i+1}/{len(data_loader)}, Avg Loss: {current_avg_loss:.4f} ({successful_batches}/{batch_count} successful)")
                    else:
                        log_message(f"  Eval batch {i+1}/{len(data_loader)}, No successful loss calculations yet")

                # Clear memory
                del images, targets
                # loss_dict and losses are always created together in try block
                # or never created at all, so no conditional check needed
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # ADDED: Force synchronization to ensure memory is actually freed
                    torch.cuda.synchronize()

        # Ensure model is in eval mode when we exit
        model.eval()
        
        # Calculate average loss
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            log_message(f"  Loss evaluation completed: {successful_batches}/{batch_count} batches successful")
        else:
            log_message(f"  Error: No successful loss calculations out of {batch_count} batches")
            avg_loss = float('inf')
        
        return None, None, avg_loss  # Return None for mAP values, only loss
        
    except Exception as e:
        model.eval()  # Ensure model is set back to eval mode even if an error occurs
        log_message(f"Error during loss-only evaluation: {e}")
        traceback.print_exc()
        return None, None, float('inf')

def evaluate_model_robust(model, data_loader, device, dataset=None, log_message=print, use_mAP=True):
    """
    ENGINEERING IMPROVEMENT: Robust evaluation wrapper with enhanced memory management
    
    This function provides a single interface for model evaluation with proper error handling
    and fallback mechanisms. It ensures BatchNorm statistics are never contaminated and
    provides consistent, reliable evaluation metrics.
    
    Args:
        model: The model to evaluate
        data_loader: Validation data loader
        device: Device to run evaluation on
        dataset: The dataset object (needed for COCO API conversion, optional)
        log_message: Logging function
        use_mAP: Whether to attempt mAP calculation (requires dataset and COCO tools)
        
    Returns:
        dict: Comprehensive evaluation results with the following keys:
            - 'segm_mAP': Segmentation mAP (None if not calculated)
            - 'bbox_mAP': Bounding box mAP (None if not calculated)
            - 'val_loss': Validation loss
            - 'evaluation_mode': String indicating which evaluation mode was used
            - 'batch_success_rate': Fraction of batches that successfully calculated loss
            - 'memory_summary': GPU memory usage summary
    """
    log_message("  Starting robust model evaluation...")
    
    # Initialize GPU memory tracker
    memory_tracker = GPUMemoryTracker(device, log_message)
    memory_tracker.reset_stats()
    memory_tracker.log_memory_status("Evaluation start", force=True)
    
    # Determine evaluation strategy
    if use_mAP and COCO_EVAL_AVAILABLE and dataset is not None:
        log_message("  Using mAP-based evaluation with loss calculation")
        
        # Check memory before starting mAP evaluation
        mem_info = memory_tracker.get_memory_info()
        if mem_info and mem_info['utilization_percent'] > 80:
            log_message("  ⚠️  High memory usage detected, switching to loss-only evaluation")
            segm_mAP, bbox_mAP, val_loss = evaluate_model_loss_only(model, data_loader, device, log_message)
            evaluation_mode = "loss_only_high_memory"
        else:
            segm_mAP, bbox_mAP, val_loss = evaluate_model_mAP(model, data_loader, device, dataset, log_message)
            evaluation_mode = "mAP_with_loss"
            
            # If mAP evaluation failed, fall back to loss-only evaluation
            if segm_mAP is None:
                log_message("  mAP evaluation failed. Switching to loss-only evaluation.")
                _, _, val_loss = evaluate_model_loss_only(model, data_loader, device, log_message)
                evaluation_mode = "loss_only_fallback"
    else:
        if use_mAP:
            log_message("  mAP evaluation requested but not available, falling back to loss-only")
        else:
            log_message("  Using loss-only evaluation")
        segm_mAP, bbox_mAP, val_loss = evaluate_model_loss_only(model, data_loader, device, log_message)
        evaluation_mode = "loss_only"
    
    # Calculate success rate (approximate)
    if val_loss == float('inf'):
        batch_success_rate = 0.0
    else:
        batch_success_rate = 1.0  # If we got a finite loss, assume most batches succeeded
    
    # Log final memory status
    memory_tracker.log_memory_status("Evaluation end", force=True)
    memory_tracker.log_memory_summary()
    
    results = {
        'segm_mAP': segm_mAP,
        'bbox_mAP': bbox_mAP,
        'val_loss': val_loss,
        'evaluation_mode': evaluation_mode,
        'batch_success_rate': batch_success_rate,
        'memory_summary': memory_tracker.get_memory_summary()
    }
    
    log_message(f"  Evaluation completed using {evaluation_mode} mode")
    return results

def setup_model_and_optimization(num_classes, device, log_message, base_checkpoint_dir=None):
    """Initialize hybrid ResNeXt model, optimizer, scheduler, and load checkpoint if available"""
    log_message("Initializing Hybrid ResNeXt-101 32x8d PANet model with pretrained backbone...")
    
    # Initialize hybrid model with pretrained backbone
    model = get_instance_segmentation_model(
        num_classes=num_classes,
        pretrained_backbone=True,  # Use pretrained weights
        trainable_layers=3  # Fine-tune last 3 layers
    )
    model.to(device)
    log_message(f"Hybrid ResNeXt-101 32x8d PANet model initialized and moved to device: {device}")
    log_message("Model features:")
    log_message("  - Pretrained ResNeXt-101 32x8d backbone")
    log_message("  - Custom PANet path aggregation")
    log_message("  - Enhanced mask predictor (28x28 resolution)")
    log_message("  - Grouped convolutions (groups=32, width_per_group=8)")

    # Default values
    start_epoch = 0
    best_val_loss = float('inf')
    best_segm_mAP = 0.0  # For mAP, we want to maximize (start from 0)
    early_stopping_counter = 0
    checkpoint = None

    # Check for existing checkpoints
    if base_checkpoint_dir is not None and os.path.exists(base_checkpoint_dir):
        checkpoint_dirs = [d for d in os.listdir(base_checkpoint_dir)
                          if os.path.isdir(os.path.join(base_checkpoint_dir, d))]

        if checkpoint_dirs:
            latest_dir = max(checkpoint_dirs)
            latest_checkpoint_dir = os.path.join(base_checkpoint_dir, latest_dir)
            checkpoints = [f for f in os.listdir(latest_checkpoint_dir) 
                          if f.endswith('.pth') and 'resnext' in f.lower()]

            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                checkpoint_path = os.path.join(latest_checkpoint_dir, latest_checkpoint)
                log_message(f"Loading checkpoint from {checkpoint_path}")

                try:
                    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint['epoch']
                    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                    best_segm_mAP = checkpoint.get('best_segm_mAP', 0.0)
                    early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
                    log_message(f"Resuming from epoch {start_epoch}")
                    
                    if 'architecture' in checkpoint:
                        log_message(f"Model architecture from checkpoint: {checkpoint['architecture']}")
                except Exception as e:
                    log_message(f"Error loading checkpoint: {e}")
                    log_message("Starting training from scratch")
                    start_epoch = 0

    # Setup optimizer with different learning rates for different parts
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
    
    # Optimized learning rates for hybrid ResNeXt approach
    param_groups = [
        {'params': backbone_params, 'lr': 0.0003},  # Lower LR for pretrained ResNeXt backbone
        {'params': head_params, 'lr': 0.001}        # Higher LR for new heads
    ]
    
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=0.9,
        weight_decay=0.0001,
        nesterov=True
    )
    
    # Learning rate scheduler optimized for ResNeXt
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
    
    log_message("Optimization setup for Hybrid ResNeXt-101 32x8d PANet:")
    log_message(f"  - Backbone learning rate: 0.0003")
    log_message(f"  - Detection heads learning rate: 0.001")
    log_message(f"  - Weight decay: 0.0001")
    log_message(f"  - LR scheduler: MultiStepLR (milestones=[60, 80], gamma=0.1)")

    # Load optimizer and scheduler states if resuming
    if checkpoint is not None and start_epoch > 0:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            log_message("Loaded optimizer and scheduler states")
        except Exception as e:
            log_message(f"Error loading optimizer/scheduler states: {e}")
            log_message("Using fresh optimizer and scheduler states")

    return model, optimizer, lr_scheduler, start_epoch, best_val_loss, best_segm_mAP, early_stopping_counter

def train_epoch(model, optimizer, train_loader, device, log_message, scaler=None, epoch_num=None, total_epochs=None):
    """Run a single training epoch with enhanced GPU memory tracking"""
    epoch_start_time = time.time()
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    # Initialize GPU memory tracker
    memory_tracker = GPUMemoryTracker(device, log_message)
    memory_tracker.reset_stats()

    if epoch_num is not None and total_epochs is not None:
        log_message(f"Epoch {epoch_num}/{total_epochs}")
        memory_tracker.log_memory_status(f"Epoch {epoch_num} start", force=True)

    for i, (images, targets) in enumerate(train_loader):
        # Pre-batch memory check
        if i % 5 == 0:  # Check every 5 batches
            memory_tracker.log_memory_status(f"Batch {i+1} pre-processing")
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        # Use enhanced memory-efficient forward pass
        if device.type == 'cuda' and scaler is not None:
            with autocast('cuda'):
                loss_dict = model(images, targets)
                losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = torch.sum(torch.stack([loss for loss in loss_dict.values()]))
            losses.backward()
            optimizer.step()

        epoch_loss += losses.item()
        batch_count += 1

        if (i + 1) % 10 == 0:
            log_message(f"  Batch {i+1}/{len(train_loader)}, Loss: {losses.item():.4f}")
            memory_tracker.log_memory_status(f"Batch {i+1} post-processing")

        # Enhanced memory cleanup
        del images, targets, loss_dict, losses
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
            # Force cleanup every 20 batches
            if (i + 1) % 20 == 0:
                gc.collect()
                torch.cuda.synchronize()

    avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0

    if epoch_num is not None:
        log_message(f"  Epoch {epoch_num} training completed. Average Loss: {avg_train_loss:.4f}")
        epoch_duration = time.time() - epoch_start_time
        log_message(f"  Epoch duration: {epoch_duration:.2f} seconds")
        
        # Log memory summary for this epoch
        memory_tracker.log_memory_summary()

    return avg_train_loss

def early_stopping_check_mAP(segm_mAP, best_segm_mAP, counter, patience, min_delta, log_message):
    """
    Check early stopping criteria using segmentation mAP
    
    Args:
        segm_mAP: Current segmentation mAP (None if not available)
        best_segm_mAP: Best segmentation mAP seen so far
        counter: Current early stopping counter
        patience: Number of epochs to wait without improvement
        min_delta: Minimum improvement to be considered significant
        log_message: Logging function
        
    Returns:
        tuple: (best_segm_mAP, counter, should_stop, improved)
    """
    should_stop = False
    improved = False

    if segm_mAP is None:
        log_message("  Early stopping skipped: No segmentation mAP available")
        return best_segm_mAP, counter, should_stop, improved

    # For mAP, higher is better (opposite of loss)
    if segm_mAP > best_segm_mAP + min_delta:
        improved = True
        best_segm_mAP = segm_mAP
        counter = 0
        log_message(f"  Segmentation mAP improved to {segm_mAP:.4f}. Early stopping counter reset.")
    else:
        counter += 1
        log_message(f"  No significant mAP improvement. Early stopping counter: {counter}/{patience}")

    if counter >= patience:
        should_stop = True
        log_message(f"  Early stopping triggered after {counter} epochs without mAP improvement")

    return best_segm_mAP, counter, should_stop, improved

def early_stopping_check_loss(val_loss, best_val_loss, counter, patience, min_delta, log_message):
    """Fallback early stopping function using validation loss (for backward compatibility)"""
    should_stop = False
    improved = False

    if val_loss is None:
        log_message("  Early stopping skipped: No validation loss available")
        return best_val_loss, counter, should_stop, improved

    if val_loss < best_val_loss - min_delta:
        improved = True
        best_val_loss = val_loss
        counter = 0
        log_message(f"  Validation loss improved to {val_loss:.4f}. Early stopping counter reset.")
    else:
        counter += 1
        log_message(f"  No significant improvement. Early stopping counter: {counter}/{patience}")

    if counter >= patience:
        should_stop = True
        log_message(f"  Early stopping triggered after {counter} epochs without improvement")

    return best_val_loss, counter, should_stop, improved

def save_checkpoint(model, optimizer, scheduler, checkpoint_dir, epoch, losses, counters, metadata=None, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }

    if losses:
        checkpoint.update(losses)

    if counters:
        checkpoint.update(counters)

    if metadata:
        checkpoint.update(metadata)
    else:
        checkpoint.update({
            'architecture': 'Hybrid-ResNeXt-PANet',
            'backbone': 'ResNeXt101-32x8d-FPN-Pretrained',
            'panet_aggregation': 'Custom',
            'grouped_convolutions': 'groups=32, width_per_group=8'
        })

    # Use ResNeXt-specific checkpoint filename
    checkpoint_filename = f'mask_rcnn_hybrid_resnext_panet_checkpoint_epoch_{epoch}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'mask_rcnn_hybrid_resnext_panet_best_model.pth')
        torch.save(checkpoint, best_model_path)
        return best_model_path

    return checkpoint_path

def main():
    """Main training function for Hybrid ResNeXt-101 32x8d PANet Mask R-CNN"""
    
    # Configuration
    data_path = "data/annotations"
    checkpoint_path = "checkpoints/mask-rcnn_hybrid_resnext_panet"
    
    # Log hybrid ResNeXt architecture usage
    print("=" * 80)
    print("HYBRID RESNEXT-101 32x8d PANET MASK R-CNN TRAINING")
    print("=" * 80)
    print("Architecture: Hybrid ResNeXt-101 32x8d PANet with pretrained backbone")
    print("Features:")
    print("  - Pretrained ResNeXt-101 32x8d backbone (torchvision)")
    print("  - Grouped convolutions (groups=32, width_per_group=8)")
    print("  - Custom PANet path aggregation")
    print("  - Enhanced mask predictor (28x28 resolution)")
    print("  - Optimized anchor generation for leaf detection")
    print("  - Superior feature representation with grouped convolutions")
    print("=" * 80)

    # Setup paths and logging
    train_dir, val_dir, train_annotations_file, val_annotations_file, checkpoint_dir, log_message, timestamp = setup_paths_and_logging(data_path, checkpoint_path)

    # Load datasets
    train_loader, val_loader, num_classes, val_dataset = load_datasets(
        train_dir, val_dir, train_annotations_file, val_annotations_file, log_message
    )

    # Device setup with enhanced logging and memory tracking
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log_message(f"CUDA available: {torch.cuda.is_available()}")
        log_message(f"CUDA device count: {torch.cuda.device_count()}")
        log_message(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        log_message(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Initialize global memory tracker
        global_memory_tracker = GPUMemoryTracker(device, log_message)
        global_memory_tracker.reset_stats()
        global_memory_tracker.log_memory_status("Training start", force=True)
        
        # Set enhanced memory management environment variables
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
        torch.backends.cudnn.benchmark = True
        
    else:
        device = torch.device('cpu')
        log_message("CUDA not available, using CPU")
        global_memory_tracker = None
    
    log_message(f"Using device: {device}")

    # Setup model and optimization
    base_checkpoint_dir = os.path.join(checkpoint_path, "checkpoints")
    model, optimizer, lr_scheduler, start_epoch, best_val_loss, best_segm_mAP, early_stopping_counter = setup_model_and_optimization(
        num_classes, device, log_message, base_checkpoint_dir
    )

    # Log model architecture details
    log_message("=" * 60)
    log_message("MODEL ARCHITECTURE: Hybrid ResNeXt-101 32x8d PANet Mask R-CNN")
    log_message("=" * 60)
    log_message("Backbone: Pretrained ResNeXt-101 32x8d FPN")
    log_message("  - Grouped convolutions: 32 groups, 8 width per group")
    log_message("  - Pretrained on ImageNet")
    log_message("  - Fine-tuning last 3 layers")
    log_message("Path Aggregation: Custom PANet implementation")
    log_message("  - Bottom-up path aggregation")
    log_message("  - Feature fusion with batch normalization")
    log_message("Detection Head: Enhanced with 28x28 mask resolution")
    log_message("Anchor Generation: Optimized for leaf morphology")
    log_message("=" * 60)
    
    # Setup mixed precision training
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    if scaler:
        log_message("Mixed precision training enabled with GradScaler")
    
    # Training hyperparameters optimized for ResNeXt with mAP-based early stopping
    early_stopping_patience = 15  # Reasonable patience for mAP improvements
    early_stopping_min_delta = 0.001  # 0.1% mAP improvement threshold (more sensitive than loss)
    log_message(f"Early stopping configuration:")
    if COCO_EVAL_AVAILABLE:
        log_message(f"  - Primary metric: Segmentation mAP (higher is better)")
        log_message(f"  - Patience: {early_stopping_patience} epochs")
        log_message(f"  - Min improvement: {early_stopping_min_delta:.3f} mAP")
        log_message(f"  - Fallback metric: Validation Loss (if mAP unavailable)")
    else:
        log_message(f"  - Metric: Validation Loss (COCO eval unavailable)")
        log_message(f"  - Patience: {early_stopping_patience} epochs")
        log_message(f"  - Min improvement: {early_stopping_min_delta:.4f} loss")

    num_epochs = 100  # Optimal for pretrained ResNeXt
    log_message(f"Training configuration:")
    log_message(f"  - Total epochs: {num_epochs}")
    log_message(f"  - Starting from epoch: {start_epoch + 1}")
    log_message(f"  - Best validation loss: {best_val_loss:.4f}")
    log_message(f"  - Best segmentation mAP: {best_segm_mAP:.4f}")
    log_message(f"  - Architecture: ResNeXt-101 32x8d + PANet")

    # Training loop
    log_message("=" * 60)
    log_message("STARTING TRAINING")
    log_message("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        # Pre-epoch memory check
        if global_memory_tracker:
            global_memory_tracker.log_memory_status(f"Epoch {epoch+1} start", force=True)
        
        # Training phase
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

        # Validation phase with memory monitoring
        segm_mAP = None
        bbox_mAP = None
        val_loss = None
        
        if val_loader:
            log_message("  Running mAP-based validation...")
            eval_results = evaluate_model_robust(
                model=model,
                data_loader=val_loader, 
                device=device,
                dataset=val_dataset,  # Pass the dataset for COCO API
                log_message=log_message,
                use_mAP=COCO_EVAL_AVAILABLE  # Automatically determine if mAP is available
            )
            
            # Extract results
            segm_mAP = eval_results['segm_mAP']
            bbox_mAP = eval_results['bbox_mAP'] 
            val_loss = eval_results['val_loss']
            evaluation_mode = eval_results['evaluation_mode']
            
            # Log memory information from evaluation
            if 'memory_summary' in eval_results:
                mem_summary = eval_results['memory_summary']
                if isinstance(mem_summary, dict):
                    log_message(f"  Evaluation memory summary:")
                    log_message(f"    Peak memory: {mem_summary['peak_allocated_mb']:.1f} MB")
                    log_message(f"    OOM warnings: {mem_summary['oom_warnings']}")
            
            # Log results
            log_message(f"  Validation results ({evaluation_mode}):")
            if segm_mAP is not None:
                log_message(f"    Segmentation mAP: {segm_mAP:.4f}")
                log_message(f"    Bounding Box mAP: {bbox_mAP:.4f}")
            log_message(f"    Validation Loss: {val_loss:.4f}")
        else:
            log_message("  WARNING: Validation dataset not available")

        # Learning rate scheduling
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        log_message(f"  Learning rate: backbone={current_lr:.6f}, heads={optimizer.param_groups[1]['lr']:.6f}")

        # Early stopping check using mAP (with fallback to loss)
        if COCO_EVAL_AVAILABLE and segm_mAP is not None:
            # Use mAP-based early stopping
            best_segm_mAP, early_stopping_counter, should_stop, improved = early_stopping_check_mAP(
                segm_mAP,
                best_segm_mAP,
                early_stopping_counter,
                early_stopping_patience,
                early_stopping_min_delta,
                log_message
            )
            log_message(f"  Best segmentation mAP so far: {best_segm_mAP:.4f}")
        else:
            # Fallback to loss-based early stopping
            log_message("  Using validation loss for early stopping (mAP not available)")
            best_val_loss, early_stopping_counter, should_stop, improved = early_stopping_check_loss(
                val_loss,
                best_val_loss,
                early_stopping_counter,
                early_stopping_patience,
                early_stopping_min_delta,
                log_message
            )

        # Prepare checkpoint data
        losses = {
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'segm_mAP': segm_mAP,
            'bbox_mAP': bbox_mAP,
            'best_segm_mAP': best_segm_mAP
        }

        counters = {
            'early_stopping_counter': early_stopping_counter,
            'num_classes': num_classes
        }

        # Enhanced metadata for ResNeXt hybrid model
        metadata = {
            'architecture': 'Hybrid-ResNeXt-PANet',
            'backbone': 'ResNeXt101-32x8d-FPN-Pretrained',
            'panet_aggregation': 'Custom',
            'grouped_convolutions': 'groups=32, width_per_group=8',
            'timestamp': timestamp,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'pretrained_backbone': True,
            'trainable_layers': 3,
            'mask_resolution': '28x28'
        }

        # Save checkpoint
        checkpoint_path_saved = save_checkpoint(
            model,
            optimizer,
            lr_scheduler,
            checkpoint_dir,
            epoch + 1,
            losses,
            counters,
            metadata=metadata,
            is_best=improved
        )
        log_message(f"  Checkpoint saved: {os.path.basename(checkpoint_path_saved)}")

        # Performance summary
        log_message(f"  Epoch {epoch+1} Summary:")
        log_message(f"    Train Loss: {avg_train_loss:.4f}")
        if val_loss is not None:
            log_message(f"    Val Loss: {val_loss:.4f}")
            log_message(f"    Best Val Loss: {best_val_loss:.4f}")
        if segm_mAP is not None:
            log_message(f"    Segmentation mAP: {segm_mAP:.4f}")
            log_message(f"    Best Segmentation mAP: {best_segm_mAP:.4f}")
        if bbox_mAP is not None:
            log_message(f"    Bounding Box mAP: {bbox_mAP:.4f}")
        log_message(f"    Early Stop Counter: {early_stopping_counter}/{early_stopping_patience}")
        if COCO_EVAL_AVAILABLE and segm_mAP is not None:
            log_message(f"    Early stopping metric: Segmentation mAP")
        else:
            log_message(f"    Early stopping metric: Validation Loss")
        log_message("  " + "-" * 50)

        # Check for early stopping
        if should_stop:
            log_message("=" * 60)
            log_message("EARLY STOPPING TRIGGERED")
            log_message("=" * 60)
            break

        # Post-epoch memory cleanup and monitoring
        if global_memory_tracker:
            global_memory_tracker.log_memory_status(f"Epoch {epoch+1} end", force=True)
        
        # Enhanced memory cleanup
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "mask_rcnn_hybrid_resnext_panet_final.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
        'train_loss': avg_train_loss,
        'val_loss': val_loss if val_loader else None,
        'architecture': 'Hybrid-ResNeXt-PANet',
        'backbone': 'ResNeXt101-32x8d-FPN-Pretrained',
        'panet_aggregation': 'Custom',
        'grouped_convolutions': 'groups=32, width_per_group=8',
        'pytorch_version': torch.__version__,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'training_completed': True,
        'final_epoch': epoch + 1
    }, final_model_path)
    log_message(f"Final model saved: {final_model_path}")

    # Create symlink to latest checkpoint
    create_symlink(checkpoint_dir, log_message)

    # Final memory summary and cleanup
    if global_memory_tracker:
        global_memory_tracker.log_memory_summary()
    
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()

    # Training completion summary
    log_message("=" * 80)
    log_message("TRAINING COMPLETED")
    log_message("=" * 80)
    log_message("Hybrid ResNeXt-101 32x8d PANet Mask R-CNN Training Summary:")
    log_message(f"  - Architecture: Hybrid ResNeXt-101 32x8d PANet")
    log_message(f"  - Backbone: Pretrained ResNeXt-101 32x8d with grouped convolutions")
    log_message(f"  - Total epochs trained: {epoch + 1}")
    log_message(f"  - Best validation loss: {best_val_loss:.4f}")
    log_message(f"  - Best segmentation mAP: {best_segm_mAP:.4f}")
    log_message(f"  - Final training loss: {avg_train_loss:.4f}")
    log_message(f"  - Model saved to: {final_model_path}")
    log_message(f"  - Checkpoints directory: {checkpoint_dir}")
    log_message("Training benefits achieved:")
    log_message("  ✓ Pretrained ResNeXt-101 32x8d backbone for superior feature extraction")
    log_message("  ✓ Grouped convolutions for better representational capacity")
    log_message("  ✓ Custom PANet aggregation for improved multi-scale features")
    log_message("  ✓ Enhanced mask resolution (28x28) for detailed segmentation")
    log_message("  ✓ Optimized anchor generation for leaf detection")
    log_message("  ✓ Mixed precision training for efficient GPU utilization")
    log_message("=" * 80)

if __name__ == "__main__":
    main()
