import copy
from collections import defaultdict
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Check if COCO evaluation tools are available
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    COCO_EVAL_AVAILABLE = True
except ImportError:
    COCO_EVAL_AVAILABLE = False
    print("Warning: pycocotools not found. COCO mAP evaluation will be unavailable.")

def get_coco_api_from_dataset(dataset):
    """
    Get COCO API instance from a torch.utils.data.Dataset object.
    This is a standard utility function from torchvision's detection references.
    
    FIXED: This function now correctly extracts the pycocotools.coco.COCO
    object, which is nested inside the torchvision.datasets.CocoDetection
    object. The original implementation was returning the CocoDetection object
    itself, causing an AttributeError during evaluation.
    """
    # The LeafDataset stores the CocoDetection object in its 'coco' attribute.
    # The CocoDetection object, in turn, stores the actual COCO API object
    # in its own 'coco' attribute.
    if hasattr(dataset, 'coco') and hasattr(dataset.coco, 'coco'):
        return dataset.coco.coco
    
    if hasattr(dataset, 'coco'):
        return dataset.coco # Fallback for other dataset structures
    
    # Fallback for datasets that don't have a 'coco' attribute
    # This might require custom adaptation based on the dataset structure
    # For now, we raise an error if 'coco' is not found.
    raise AttributeError("The provided dataset does not have a 'coco' attribute. "
                         "Please ensure your dataset is compatible with COCO evaluation.")

class CocoEvaluator:
    """
    A utility class to handle COCO evaluation.
    This is a standard utility class from torchvision's detection references.
    """
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            self.coco_eval[iou_type] = COCOeval(self.coco_gt, coco_dt, iou_type)
            self.coco_eval[iou_type].params.imgIds = self.img_ids

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        raise ValueError(f"iou_type {iou_type} not supported")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = self.convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend([{
                "image_id": original_id,
                "category_id": labels[k],
                "bbox": box,
                "score": scores[k],
            } for k, box in enumerate(boxes)])
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]
            masks = masks > 0.5
            rles = [self.mask_to_rle(mask) for mask in masks]
            coco_results.extend([{
                "image_id": original_id,
                "category_id": labels[k].item(),
                "segmentation": rle,
                "score": scores[k].item(),
            } for k, rle in enumerate(rles)])
        return coco_results

    def convert_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

    def mask_to_rle(self, mask):
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(mask.cpu().numpy()))
        if isinstance(rle['counts'], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    def synchronize_between_processes(self):
        # This is a placeholder for distributed training.
        # In a single-process setup, this does nothing.
        pass

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.evaluate()
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize() 
