import os
import json
import subprocess
import optuna


def objective(trial):
    # Suggest hyperparameters
    params = {
        'lr0': trial.suggest_float('lr0', 1e-5, 1e-2, log=True),
        'batch_size': trial.suggest_int('batch_size', 4, 16),
        'optimizer': trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW']),
        'dropout': trial.suggest_float('dropout', 0.0, 0.5),
        'attention': trial.suggest_categorical('attention', ['none', 'sca', 'eca']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    }

    # Construct output directory for this trial
    trial_dir = f"optuna_trials/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)

    # Build command to run training script
    command = [
        "python", "train_val_yolo11x-seg.py",
        "--data", "../../data/annotations/yolo/640/data.yaml",
        "--output-dir", trial_dir,
        "--model", "yolo11x-seg.pt",
        "--epochs", "100",
        "--batch-size", str(params['batch_size']),
        "--optimizer", params['optimizer'],
        "--attention", params['attention'],
        "--dropout", str(params['dropout']),
        "--rect",
        "--device", "0"
    ]

    # Run training
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        return 0.0

    # Read metrics
    metrics_path = os.path.join(trial_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        return 0.0

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Determine which metric is the primary metric, then utilize it as the objective
    # primary_metric = metrics.get("primary_metric")

    return metrics.get("box_map", 0.0)

def main():
    os.makedirs("optuna_trials", exist_ok=True)

    study = optuna.create_study(
        study_name="yolo11x_seg_optimization_640",
        storage="sqlite:///yolo11x_seg_optimization.db",
        load_if_exists=True,
        direction="maximize"
    )

    study.optimize(objective, n_trials=100)

    # Save best trial
    with open("best_trial.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)


if __name__ == "__main__":
    main()
