#!/usr/bin/env python3
"""
train_yolo.py — Train YOLO detector on ancient rubbing character dataset.

Tuned for: V100-32GB (single GPU by default to avoid DDP frozen-layer crash).
For dual GPU, pass --device '0,1' --batch 16, but note YOLO11m freezes DFL
layer which can crash DDP. Single GPU is recommended.

Pretrained weights handling (server is offline):
    1. Download weights on a machine with internet:
         wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt
    2. Place in project: models/yolo11m.pt
    3. The script will use them automatically.
    If no weights found, falls back to training from scratch.

Usage:
    python scripts/train_yolo.py \
        --data /home/apulis-dev/userdata/lbh/danc/yolo_dataset/dataset.yaml \
        --weights models/yolo11m.pt \
        --epochs 150 \
        --imgsz 1280 \
        --batch 16 \
        --project /home/apulis-dev/userdata/lbh/danc/runs/detect
"""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO detector for ancient char detection")
    parser.add_argument("--data", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/yolo_dataset/dataset.yaml")
    parser.add_argument("--weights", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/DanC/model/yolo11m.pt",
                        help="Path to pretrained weights.")
    parser.add_argument("--model_cfg", type=str, default="yolo11m.yaml",
                        help="Model config YAML (used when training from scratch)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="Training image size (1280 recommended for rubbing images)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size. For single V100-32GB with imgsz=1280, use 8-12.")
    parser.add_argument("--device", type=str, default="0",
                        help="GPU device. Use '0' for single GPU (avoids DDP issues with frozen layers). "
                             "Only use '0,1' if batch>=16 and you've verified DDP compatibility.")
    parser.add_argument("--project", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/runs/detect")
    parser.add_argument("--name", type=str, default="ancient_char_det")
    parser.add_argument("--resume", type=str, default="",
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights_path = Path(args.weights)

    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    elif weights_path.exists():
        print(f"Loading pretrained weights: {weights_path}")
        model = YOLO(str(weights_path))
    else:
        print(f"Weights not found at {weights_path}")
        print(f"Training from scratch with config: {args.model_cfg}")
        model = YOLO(args.model_cfg)

    train_args = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,

        # --- Optimizer ---
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        warmup_momentum=0.8,

        # --- Augmentation (tuned for document/rubbing images) ---
        # Reduce mosaic — too aggressive for structured document layout
        mosaic=0.5,
        # Disable mixup — meaningless for OCR
        mixup=0.0,
        # Small rotation only — characters have fixed orientation
        degrees=5.0,
        # Moderate scale variation
        scale=0.3,
        # Slight translation
        translate=0.1,
        # No flipping — characters are NOT flip-invariant
        flipud=0.0,
        fliplr=0.0,
        # Color/brightness augmentation — useful for rubbing image variation
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,

        # --- Training stability ---
        patience=30,
        save_period=10,
        workers=8,
        amp=True,
        cos_lr=True,
        close_mosaic=20,

        # --- Validation ---
        val=True,
        plots=True,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"Training config:")
    print(f"  Data:    {args.data}")
    print(f"  Model:   {weights_path if weights_path.exists() else args.model_cfg}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  ImgSize: {args.imgsz}")
    print(f"  Batch:   {args.batch}")
    print(f"  Device:  {args.device}")
    print(f"  Output:  {args.project}/{args.name}")
    print(f"{'='*60}\n")

    results = model.train(**train_args)

    print(f"\nTraining complete.")
    print(f"Best model: {args.project}/{args.name}/weights/best.pt")

    metrics = model.val()
    print(f"\nValidation results:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall:    {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
