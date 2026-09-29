"""
=============================================================================
IBVAP - Intelligent Border Video Analytics Platform
Module: Custom YOLO License Plate Model Training with MLflow & DVC
File: models/anpr/plate_detector.py
=============================================================================
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
import mlflow
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def train_plate_model(
    data_yaml: str = "configs/anpr_data.yaml",
    base_model: str = "models/detection/yolo11n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    experiment_name: str = "IBVAP_ANPR_Plate_Detector",
    output_model_dir: str = "models/anpr",
):
    """
    Train custom YOLO license plate detector and log full experiment telemetry to MLflow.
    """
    # 1. Resolve configuration and model paths
    yaml_path = Path(data_yaml)
    if not yaml_path.is_absolute():
        yaml_path = PROJECT_ROOT / yaml_path

    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset config not found at: {yaml_path}")

    base_model_path = Path(base_model)
    if not base_model_path.is_absolute():
        base_model_path = PROJECT_ROOT / base_model_path

    out_dir = Path(output_model_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup MLflow Tracking (Using SQLite backend)
    db_path = PROJECT_ROOT / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path.as_posix()}"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    print("=" * 75)
    print(" 🛡️  IBVAP - ANPR License Plate Detector Training")
    print("=" * 75)
    print(f" • Dataset Config: {yaml_path}")
    print(f" • Base Model:     {base_model_path}")
    print(f" • Epochs:         {epochs}")
    print(f" • Image Size:     {imgsz}")
    print(f" • Batch Size:     {batch_size}")
    print(f" • Learning Rate:  {learning_rate}")
    print(f" • MLflow DB:      {db_path.as_posix()}")
    print("=" * 75)

    with mlflow.start_run(run_name=f"yolo11n_plate_ep{epochs}_lr{learning_rate}") as run:
        # Log Hyperparameters
        mlflow.log_params({
            "model_architecture": "yolo11n",
            "dataset_config": str(yaml_path.name),
            "epochs": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "initial_lr": learning_rate,
            "optimizer": "auto",
        })

        # Load Base Model
        model = YOLO(str(base_model_path))

        # Execute YOLO Training
        # NOTE: workers=0 required on Windows to avoid DataLoader multiprocessing crash
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            lr0=learning_rate,
            project=str(PROJECT_ROOT / "runs" / "anpr_train"),
            name="yolo11_plate_run",
            exist_ok=True,
            verbose=True,
            workers=0,
        )

        # Log Metrics to MLflow and export to metrics.json for DVC
        exported_metrics = {}
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    clean_k = k.replace("(", "").replace(")", "").replace("/", "_")
                    mlflow.log_metric(clean_k, float(v))
                    exported_metrics[clean_k] = round(float(v), 5)

        # Write metrics.json for DVC experiments table
        metrics_file = PROJECT_ROOT / "metrics.json"
        import json
        with open(metrics_file, "w") as f:
            json.dump(exported_metrics, f, indent=2)
        print(f"[IBVAP-Training] Saved DVC metrics to: {metrics_file}")

        # Copy best.pt to models/anpr/plate_detector.pt
        best_pt_src = PROJECT_ROOT / "runs" / "anpr_train" / "yolo11_plate_run" / "weights" / "best.pt"
        target_pt = out_dir / "plate_detector.pt"

        if best_pt_src.exists():
            shutil.copy(str(best_pt_src), str(target_pt))
            print(f"\n[IBVAP-Training] Copied best checkpoint to: {target_pt}")
            mlflow.log_artifact(str(target_pt), artifact_path="model_weights")

        print(f"\n✅ MLflow Run ID: {run.info.run_id}")
        print("📁 Training artifacts recorded successfully in MLflow!\n")

    return str(target_pt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom YOLO license plate detector with MLflow")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    args = parser.parse_args()

    train_plate_model(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        learning_rate=args.lr,
    )
