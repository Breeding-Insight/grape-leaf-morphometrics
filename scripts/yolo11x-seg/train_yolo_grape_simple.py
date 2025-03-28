from ultralytics import YOLO
import torch
import os

def main():
    # Automatic download (official Ultralytics assets)
    model = YOLO("yolo11x-seg.pt")

    # torch clear gpu cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        torch.mps.empty_cache()

    # Start training
    model.train(
        data="/Users/aja294/Documents/Hemp_local/projects/yoloV8_marker_detect/yolo_versions/metrics_marker_detection.v3i.yolov11/data.yaml",
        epochs=100,
        batch=4,
        imgsz=640,
        device="mps",  # Use CUDA for GPU acceleration
        workers=0,  # Critical for macOS stability
        amp=False,
        nms=False,
        resume=True
    )

    # Save final model
    model.save("/Users/aja294/Documents/Hemp_local/projects/yolo11_marker_detect/runs/detect/train22/weights/final.pt")

if __name__ == "__main__":
    main()
