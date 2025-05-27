import os
import argparse
import yaml
import json
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Mask R-CNN on custom dataset"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Override output directory"
    )
    # Make test dataset paths optional, config file is preferred
    parser.add_argument(
        "--test_json", type=str, 
        help="Optional: Override test annotations JSON from config"
    )
    parser.add_argument(
        "--test_img_dir", type=str,
        help="Optional: Override test images directory from config"
    )
    return parser.parse_args()


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config_data = load_config(args.config)

    project_root = os.getcwd()
    
    # Set up datasets
    train_json = os.path.join(
        project_root, config_data["datasets"]["train_json"]
    )
    train_img_dir = os.path.join(
        project_root, config_data["datasets"]["train_img_dir"]
    )
    val_json = os.path.join(
        project_root, config_data["datasets"]["val_json"]
    )
    val_img_dir = os.path.join(
        project_root, config_data["datasets"]["val_img_dir"]
    )
    
    # Define test dataset - prefer config file over CLI args
    test_json = args.test_json if args.test_json else os.path.join(
        project_root, config_data["datasets"].get("test_json", "")
    )
    test_img_dir = args.test_img_dir if args.test_img_dir else os.path.join(
        project_root, config_data["datasets"].get("test_img_dir", "")
    )

    TRAIN_NAME = "custom_train"
    VAL_NAME = "custom_val"
    TEST_NAME = "custom_test"

    if TRAIN_NAME not in DatasetCatalog.list():
        register_coco_instances(
            TRAIN_NAME, {}, train_json, train_img_dir
        )
    if VAL_NAME not in DatasetCatalog.list():
        register_coco_instances(
            VAL_NAME, {}, val_json, val_img_dir
        )
    if TEST_NAME not in DatasetCatalog.list() and test_json and test_img_dir:
        register_coco_instances(
            TEST_NAME, {}, test_json, test_img_dir
        )

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(config_data["model_zoo_config"])
    )
    
    # Use test dataset if available
    if test_json and test_img_dir:
        cfg.DATASETS.TEST = (TEST_NAME,)
        eval_dataset = TEST_NAME
    else:
        cfg.DATASETS.TEST = (VAL_NAME,)
        eval_dataset = VAL_NAME
    
    # Explicitly set training dataset to prevent default COCO dataset loading
    cfg.DATASETS.TRAIN = (TRAIN_NAME,)
    
    cfg.DATALOADER.NUM_WORKERS = config_data["dataloader"]["num_workers"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = config_data["roi_heads"]["num_classes"]
    output_dir = (
        args.output_dir if args.output_dir else config_data["output_dir"]
    )
    cfg.OUTPUT_DIR = output_dir

    # Use specified model path from config
    model_path = config_data.get(
        "model_path",
        os.path.join(config_data["output_dir"], "model_final.pth")
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}"
        )
    cfg.MODEL.WEIGHTS = model_path

    # Build model and evaluator
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    evaluator = COCOEvaluator(
        eval_dataset, cfg, False, output_dir=cfg.OUTPUT_DIR
    )
    test_loader = build_detection_test_loader(cfg, eval_dataset)
    metrics = inference_on_dataset(trainer.model, test_loader, evaluator)

    # Save metrics to JSON file (from config)
    metrics_output_path = os.path.join(
        cfg.OUTPUT_DIR,
        config_data.get("metrics_output", "metrics.json")
    )
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    with open(metrics_output_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print mAP and AP at specific IoUs for bbox and segm
    for task in ["bbox", "segm"]:
        if task in metrics:
            print(f"\n===== {task.upper()} COCO Evaluation Metrics =====")
            print(
                f"mAP (AP 50-95): {metrics[task].get('AP', 'N/A'):.3f}"
            )
            print(
                f"AP@IoU=0.50: {metrics[task].get('AP50', 'N/A'):.3f}"
            )
            print(
                f"AP@IoU=0.75: {metrics[task].get('AP75', 'N/A'):.3f}"
            )
            # Extract AP@IoU=0.95 from the raw COCOeval results
            if 'AP95' in metrics[task]:
                print(
                    f"AP@IoU=0.95: {metrics[task].get('AP95', 'N/A'):.3f}"
                )
            else:
                # Try to get it from the raw results
                raw_metrics = metrics[task].get('raw_metrics', {})
                if 'AP' in raw_metrics:
                    ap_values = raw_metrics['AP']
                    if isinstance(ap_values, list) and len(ap_values) >= 10:
                        print(
                            f"AP@IoU=0.95: {ap_values[9]:.3f}"
                        )
        else:
            print(f"No {task} results found in metrics.")


if __name__ == "__main__":
    main()