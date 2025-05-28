# YOLOv11x-seg Environment for Grape Peduncle Segmentation

This directory contains environment configurations for training YOLOv11x-seg models on grape peduncle segmentation tasks.

## Environment Files

- `yolox11seg-env.yaml`: Original environment with Ultralytics 8.x support
- `yolox11seg-env-v11.yaml`: Updated environment with Ultralytics 11.x support for YOLOv11 features

## Setting Up the YOLOv11 Environment

### Creating the Environment

```bash
# Create the environment from the YAML file
conda env create -f environments/yolo11x_seg/yolox11seg-env-v11.yaml
```

### Activating the Environment

```bash
# Activate the environment
conda activate yolo11xseg-env-v11
```

### Updating the Environment

If you need to update packages in the environment:

```bash
conda env update -f environments/yolo11x_seg/yolox11seg-env-v11.yaml --prune
```

## Using with SLURM

For training on a cluster with SLURM, a batch script is provided:

```bash
# Submit the SLURM job
sbatch scripts/yolo11xseg/slurm/train_yolo11_peduncle_v11.sh
```

## Running Training Locally

To run training locally (if you have sufficient GPU resources):

```bash
# Activate the environment
conda activate yolo11xseg-env-v11

# Run the training script with the config file
python scripts/yolo11xseg/train_validate/train_yolo11_peduncle.py \
  --config scripts/yolo11xseg/configs/yolox11seg-peduncle.yaml
```

## YOLOv11 Features

The YOLOv11 version includes support for:

- `attention`: Different attention mechanisms including SCA (Spatial Channel Attention)
- `repvgg_block`: RepVGG blocks for efficient inference
- `mask`: Improved segmentation mask handling
- `lr_dropout`: Learning rate dropout for improved training stability

## Troubleshooting

If you encounter issues with missing features:

1. Confirm that the Ultralytics version is 11.x:
   ```bash
   pip show ultralytics
   ```

2. Check for GPU compatibility:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Verify that the CUDA version matches what PyTorch expects:
   ```bash
   python -c "import torch; print('CUDA:', torch.version.cuda)"
   ``` 