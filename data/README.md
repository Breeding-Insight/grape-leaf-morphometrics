# Grape Leaf Metrics

This repository contains code for analyzing grape leaf metrics using computer vision and deep learning techniques. All data, models, and configurations are stored on the Cornell BioHPC server.

## Repository Structure

The project is organized as follows:

```
grape_leaf_metrics/
├── configs/                 # Configuration files
│   └── torch_nightly_cuda12.6-env.yaml  # Conda environment configuration
│
├── data/                    # Data storage
│   ├── annotations/         # Annotation files for training models
│   │   ├── coco/            # COCO format annotations
│   │   └── yolov11n-seg/    # YOLOv11n segmentation format annotations
│   ├── processed/           # Processed data files
│   └── raw/                 # Raw data files
│       └── images/          # Original leaf images
│
├── models/                  # Trained models
│   └── mask_rcnn/           # Mask R-CNN model files
│       └── checkpoints_*/   # Model checkpoints from various training runs
│
└── scripts/                 # Analysis and utility scripts
```

## Data Storage

All data for this project is stored on the Cornell BioHPC server and is not included in the GitHub repository. The data is located at:

```
/workdir/data/grape/grape_pheno/grape_leaf_metrics/
```

### Data Access

To access the data, you need an account on the Cornell BioHPC server. There are several methods to access the data:

#### 1. Direct Server Access

```bash
# SSH into the server
ssh username@cbsubi2.biohpc.cornell.edu

# Navigate to the project directory
cd /workdir/data/grape/grape_pheno/grape_leaf_metrics/
```

#### 2. SSHFS Mount

You can mount the remote directory to your local machine:

```bash
# Create a mount point
mkdir -p ~/server_mounts/grape_leaf_metrics

# Mount the remote directory
sshfs username@cbsubi2.biohpc.cornell.edu:/workdir/data/grape/grape_pheno/grape_leaf_metrics ~/server_mounts/grape_leaf_metrics
```

#### 3. SFTP Access

For transferring files:

```bash
sftp username@cbsubi2.biohpc.cornell.edu:/workdir/data/grape/grape_pheno/grape_leaf_metrics
```

## Data Organization

### Raw Data

The raw data consists of grape leaf images stored in:
```
/workdir/data/grape/grape_pheno/grape_leaf_metrics/data/raw/images/
```

### Annotations

Annotations are available in multiple formats:

1. **COCO Format**: Used for Mask R-CNN and other instance segmentation models
   ```
   /workdir/data/grape/grape_pheno/grape_leaf_metrics/data/annotations/coco/
   ```

2. **YOLOv11n-seg Format**: Used for YOLO-based segmentation models
   ```
   /workdir/data/grape/grape_pheno/grape_leaf_metrics/data/annotations/yolov11n-seg/
   ```

### Models

Trained models are stored in the models directory:

```
/workdir/data/grape/grape_pheno/grape_leaf_metrics/models/
```

Currently, we have Mask R-CNN models with multiple checkpoint directories representing different training runs. Each checkpoint directory contains model weights and training logs.

## Environment Setup

To set up the required environment, use the provided conda environment file:

```bash
# SSH into the server
ssh username@cbsubi2.biohpc.cornell.edu

# Navigate to the project directory
cd /workdir/data/grape/grape_pheno/grape_leaf_metrics/

# Create conda environment
conda env create -f configs/torch_nightly_cuda12.6-env.yaml

# Activate the environment
conda activate grape-leaf-metrics
```

## Running the Code

To run the code in this repository, you'll need to:

1. Clone this GitHub repository
2. Access the data on the Cornell BioHPC server
3. Update the data paths in the configuration files to point to your data location

Example:

```bash
# Clone the repository
git clone https://github.com/username/grape-leaf-metrics.git
cd grape-leaf-metrics

# Set up data paths
export DATA_ROOT=/path/to/mounted/data
# OR
export DATA_ROOT=/workdir/data/grape/grape_pheno/grape_leaf_metrics/

# Run a script
python scripts/train_model.py --config configs/model_config.yaml
```

## Model Checkpoints

The model checkpoints are organized by timestamp in the format ```checkpoints_YYYYMMDD_HHMMSS```. Each checkpoint directory contains:

- Model weights (.pth files)
- Training configuration
- Training and validation logs
- Evaluation metrics

To use a specific checkpoint:

```python
from models.mask_rcnn import load_model

# Load a specific checkpoint
model = load_model("/path/to/models/mask_rcnn/checkpoints_20250306_140732/model_final.pth")
```

## Contributing

When contributing to this repository, please note that the data should remain on the server and not be committed to GitHub. Update paths in your code to reference the server location.

## Contact

For questions about this project or access to the data, please contact the project maintainers.

---

**Note**: This README provides an overview of the project structure and data organization. For detailed usage instructions, please refer to the documentation in the specific script files.

