"""
EchoPath - YOLO Model Training Script (Phase 1: Object Detection)

NOTE ON MODEL TRAINING LOCATION:
- Heavy model training should preferably be executed in Google Colab or a cloud environment with a dedicated GPU.
- Lightweight debugging and small test runs (few epochs/small dataset) can be executed locally on Apple Silicon (MPS) or CPU.

This script configures and launches Ultralytics YOLO object detection training.
Supported models include YOLO11n (`yolo11n.pt`), YOLOv10n (`yolov10n.pt`), or YOLOv8n (`yolov8n.pt`).

Usage example (Cloud / Colab GPU):
    python models/detection/train_yolo.py --data data/mobility_coco.yaml --model yolo11n.pt --epochs 100 --batch 32 --device 0

Usage example (Local Apple Silicon / CPU test):
    python models/detection/train_yolo.py --data data/mobility_coco.yaml --model yolo11n.pt --epochs 2 --batch 4 --device mps
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def resolve_device(device: str) -> str:
    """Resolves 'auto' device selection to PyTorch compatible device string ('0', 'mps', or 'cpu')."""
    if not device or device == "auto":
        if torch.cuda.is_available():
            return "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Ultralytics YOLO model for EchoPath Object Detection."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/mobility_coco.yaml",
        help="Path to dataset configuration YAML file (default: data/mobility_coco.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base model weight file or architectural configuration (default: yolo11n.pt; options: yolo11n.pt, yolov10n.pt, etc.)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image resolution in pixels (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size per training step (default: 16)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device ('auto', 'cpu', 'mps', or CUDA index e.g., '0') (default: auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Directory to save training logs and checkpoints (default: runs/detect)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="echopath_yolo",
        help="Experiment run name (default: echopath_yolo)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loader worker threads (default: 4)",
    )
    return parser.parse_args()


def train(args):
    """Initializes YOLO model and begins training loop."""
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset configuration file not found at '{args.data}'.")
        print("Please prepare the dataset configuration YAML before running training.")
        sys.exit(1)

    target_device = resolve_device(args.device)

    print(f"=== EchoPath YOLO Training Initializing ===")
    print(f"Model Architecture : {args.model}")
    print(f"Dataset YAML       : {args.data}")
    print(f"Epochs             : {args.epochs}")
    print(f"Image Size         : {args.imgsz}")
    print(f"Batch Size         : {args.batch}")
    print(f"Device             : {target_device}")
    print(f"Output Project Dir : {args.project}/{args.name}")
    print("==========================================")

    model = YOLO(args.model)

    # Launch Ultralytics training workflow
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=target_device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        exist_ok=True,
    )

    print(f"\nTraining session completed. Checkpoints and logs saved to: {Path(args.project) / args.name}")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
