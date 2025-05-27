import os
import argparse
import yaml
from pathlib import Path
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg, CfgNode as CN
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
import threading
import time
import subprocess

def monitor_gpu(interval, stop_event):
    while not stop_event.is_set():
        subprocess.run(["nvidia-smi"])
        time.sleep(interval)

def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on custom dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--resume", action="store_true", help="Resume training if checkpoint exists")
    return parser.parse_args()

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

args = parse_args()
config_data = load_config(args.config)

# Instead of hardcoding the path resolution method
# Use the current working directory which SLURM sets correctly
project_root = os.getcwd()  # This will be '/workdir/data/grape/grape_pheno/grape-leaf-morphometrics/'

# Then use this to resolve your paths
train_json = os.path.join(project_root, config_data["datasets"]["train_json"])
train_img_dir = os.path.join(project_root, config_data["datasets"]["train_img_dir"])
val_json = os.path.join(project_root, config_data["datasets"]["val_json"])
val_img_dir = os.path.join(project_root, config_data["datasets"]["val_img_dir"])

# Dataset names
TRAIN_NAME = "custom_train"
VAL_NAME = "custom_val"

# Only register if not already done
if TRAIN_NAME not in DatasetCatalog.list():
    register_coco_instances(TRAIN_NAME, {}, train_json, train_img_dir)

if VAL_NAME not in DatasetCatalog.list():
    register_coco_instances(VAL_NAME, {}, val_json, val_img_dir)

# Config setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(config_data["model_zoo_config"]))

cfg.DATASETS.TRAIN = (TRAIN_NAME,)
cfg.DATASETS.TEST = (VAL_NAME,)
cfg.DATALOADER.NUM_WORKERS = config_data["dataloader"]["num_workers"]

cfg.SOLVER.IMS_PER_BATCH = config_data["solver"]["ims_per_batch"]
cfg.SOLVER.BASE_LR = config_data["solver"]["base_lr"]
cfg.SOLVER.MAX_ITER = config_data["solver"]["max_iter"]
cfg.SOLVER.STEPS = tuple(config_data["solver"]["steps"])

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = config_data["roi_heads"]["batch_size_per_image"]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = config_data["roi_heads"]["num_classes"]

# Apply custom anchor settings if specified in the config
if "model" in config_data and "anchor_generator" in config_data["model"]:
    if "sizes" in config_data["model"]["anchor_generator"]:
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = config_data["model"]["anchor_generator"]["sizes"]
    if "aspect_ratios" in config_data["model"]["anchor_generator"]:
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = config_data["model"]["anchor_generator"]["aspect_ratios"]
    print(f"Using custom anchor sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"Using custom aspect ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")

# Apply custom input size settings if specified in the config
if "INPUT" in config_data:
    if "MIN_SIZE_TRAIN" in config_data["INPUT"]:
        # Handle MIN_SIZE_TRAIN properly - convert string to list of integers if needed
        if isinstance(config_data["INPUT"]["MIN_SIZE_TRAIN"], str):
            # Clean the string by removing parentheses and other non-numeric characters except commas
            clean_str = config_data["INPUT"]["MIN_SIZE_TRAIN"].replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            # Split string by comma and convert each part to integer
            min_sizes = [int(size.strip()) for size in clean_str.split(",")]
            cfg.INPUT.MIN_SIZE_TRAIN = min_sizes
        else:
            cfg.INPUT.MIN_SIZE_TRAIN = config_data["INPUT"]["MIN_SIZE_TRAIN"]
        print(f"Using custom MIN_SIZE_TRAIN: {cfg.INPUT.MIN_SIZE_TRAIN}")
    if "MAX_SIZE_TRAIN" in config_data["INPUT"]:
        cfg.INPUT.MAX_SIZE_TRAIN = config_data["INPUT"]["MAX_SIZE_TRAIN"]
        print(f"Using custom MAX_SIZE_TRAIN: {cfg.INPUT.MAX_SIZE_TRAIN}")
    if "MIN_SIZE_TEST" in config_data["INPUT"]:
        # Handle MIN_SIZE_TEST similarly
        if isinstance(config_data["INPUT"]["MIN_SIZE_TEST"], str):
            # Clean the string by removing parentheses and other non-numeric characters except commas
            clean_str = config_data["INPUT"]["MIN_SIZE_TEST"].replace('(', '').replace(')', '').replace('[', '').replace(']', '')
            # Split string by comma and convert each part to integer
            min_sizes = [int(size.strip()) for size in clean_str.split(",")]
            cfg.INPUT.MIN_SIZE_TEST = min_sizes
        else:
            cfg.INPUT.MIN_SIZE_TEST = config_data["INPUT"]["MIN_SIZE_TEST"]
        print(f"Using custom MIN_SIZE_TEST: {cfg.INPUT.MIN_SIZE_TEST}")
    if "MAX_SIZE_TEST" in config_data["INPUT"]:
        cfg.INPUT.MAX_SIZE_TEST = config_data["INPUT"]["MAX_SIZE_TEST"]
        print(f"Using custom MAX_SIZE_TEST: {cfg.INPUT.MAX_SIZE_TEST}")

# Apply custom ROI heads settings for better single object detection
if "roi_heads" in config_data:
    if "SCORE_THRESH_TEST" in config_data["roi_heads"]:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config_data["roi_heads"]["SCORE_THRESH_TEST"]
        print(f"Using custom SCORE_THRESH_TEST: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    if "NMS_THRESH_TEST" in config_data["roi_heads"]:
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = config_data["roi_heads"]["NMS_THRESH_TEST"]
        print(f"Using custom NMS_THRESH_TEST: {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}")
    if "DETECTIONS_PER_IMAGE" in config_data["roi_heads"]:
        cfg.TEST.DETECTIONS_PER_IMAGE = config_data["roi_heads"]["DETECTIONS_PER_IMAGE"]
        print(f"Using custom DETECTIONS_PER_IMAGE: {cfg.TEST.DETECTIONS_PER_IMAGE}")

# Configure mask-specific parameters for instance segmentation
if "mask_head" in config_data:
    mask_config = config_data["mask_head"]
    if "NUM_CONV" in mask_config:
        cfg.MODEL.ROI_MASK_HEAD.NUM_CONV = mask_config["NUM_CONV"]
        print(f"Using custom mask NUM_CONV: {cfg.MODEL.ROI_MASK_HEAD.NUM_CONV}")
    if "POOLER_RESOLUTION" in mask_config:
        cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = mask_config["POOLER_RESOLUTION"]
        print(f"Using custom mask POOLER_RESOLUTION: {cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION}")
    if "POOLER_SAMPLING_RATIO" in mask_config:
        cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = mask_config["POOLER_SAMPLING_RATIO"]
        print(f"Using custom mask POOLER_SAMPLING_RATIO: {cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO}")
    if "LOSS_WEIGHT" in mask_config:
        cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = mask_config["LOSS_WEIGHT"]
        print(f"Using custom mask LOSS_WEIGHT: {cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT}")

# For data augmentation
if "augmentation" in config_data:
    aug_config = config_data["augmentation"]
    if "random_flip" in aug_config:
        cfg.INPUT.RANDOM_FLIP = aug_config["random_flip"]
    # Set up color augmentation
    cfg.INPUT.FORMAT = "BGR"
    if any(k in aug_config for k in ["brightness", "contrast", "saturation", "hue"]):
        augmentations = []
        if "brightness" in aug_config:
            cfg.INPUT.BRIGHTNESS = aug_config["brightness"]
            print(f"Using brightness augmentation: {cfg.INPUT.BRIGHTNESS}")
        if "contrast" in aug_config:
            cfg.INPUT.CONTRAST = aug_config["contrast"]
            print(f"Using contrast augmentation: {cfg.INPUT.CONTRAST}")
        if "saturation" in aug_config:
            cfg.INPUT.SATURATION = aug_config["saturation"]
            print(f"Using saturation augmentation: {cfg.INPUT.SATURATION}")
        if "hue" in aug_config:
            cfg.INPUT.HUE = aug_config["hue"]
            print(f"Using hue augmentation: {cfg.INPUT.HUE}")

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_data["model_zoo_config"])

# Provide information about the shape mismatches
print("\nNOTE: Shape mismatches are expected when using a pre-trained model with a different")
print("      number of classes or anchor configurations. The following differences are normal:")
print("      - proposal_generator.rpn_head: Differs due to custom anchor configurations")
print("      - roi_heads.box_predictor: Differs due to different number of classes")
print("      - roi_heads.mask_head: Differs due to different number of classes for segmentation\n")

output_dir = args.output_dir if args.output_dir else config_data["output_dir"]
cfg.OUTPUT_DIR = output_dir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Trainer
trainer = DefaultTrainer(cfg)

# Set checkpoint loading behavior to ignore shape mismatches
if config_data.get("ignore_shape_mismatch", True):
    print("INFO: Shape mismatches between the pre-trained model and current model will be ignored.")
    print("      This is expected when fine-tuning with different classes or anchor configurations.")
    # Apply this for fine-tuning from pre-trained models with different class counts
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a testing threshold

gpu_logging = config_data.get("logging", {}).get("gpu_monitoring", False)
if gpu_logging:
    interval = config_data["logging"].get("monitor_interval_sec", 10)
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpu, args=(interval, stop_event))
    monitor_thread.start()

trainer.resume_or_load(resume=args.resume)
trainer.train()

if gpu_logging:
    stop_event.set()
    monitor_thread.join()

# Evaluation after training
evaluator = COCOEvaluator(VAL_NAME, cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, VAL_NAME)
metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
print("== Evaluation Metrics:")
print(metrics)
