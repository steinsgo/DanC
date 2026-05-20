#!/usr/bin/env python3
"""
run_inference.py — End-to-end inference pipeline for ancient character OCR.

Stage 1: YOLO detector finds character bounding boxes.
Stage 2: ResNet classifier recognizes each cropped character.

Output: /saisresult/prediction.json

Usage (inside Docker):
    python /app/src/run_inference.py

Usage (local testing):
    python scripts/run_inference.py \
        --input_dir /saisdata/13/eval/images \
        --det_model /path/to/best_det.pt \
        --rec_model /path/to/best_rec.pt \
        --char_dict /path/to/char_dict.json \
        --output /saisresult/prediction.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models


def load_detector(det_model_path: str, device: str):
    from ultralytics import YOLO
    model = YOLO(det_model_path)
    return model


def load_recognizer(rec_model_path: str, device: str):
    ckpt = torch.load(rec_model_path, map_location=device, weights_only=False)

    num_classes = ckpt["num_classes"]
    backbone = ckpt.get("backbone", "resnet50")
    img_size = ckpt.get("img_size", 64)
    class_to_idx = ckpt.get("class_to_idx", {})

    if backbone == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, img_size, idx_to_class


def build_rec_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def run_detection(detector, img_path: str, conf_thresh: float = 0.25,
                   iou_thresh: float = 0.45, imgsz: int = 1280,
                   device: str = "cuda:0"):
    results = detector.predict(
        source=img_path,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        verbose=False,
        device=device,
    )
    boxes = []
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()
                x = int(round(x1))
                y = int(round(y1))
                w = int(round(x2 - x1))
                h = int(round(y2 - y1))
                boxes.append({"bbox": [x, y, w, h], "conf": conf})
    return boxes


def run_recognition(recognizer, rec_transform, img: Image.Image,
                    boxes: list, idx_to_class: dict, id_to_char: dict,
                    device: str, batch_size: int = 128):
    if not boxes:
        return []

    img_w, img_h = img.size
    crops = []
    valid_indices = []

    for i, box_info in enumerate(boxes):
        x, y, w, h = box_info["bbox"]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue

        crop = img.crop((x1, y1, x2, y2))
        crop_tensor = rec_transform(crop)
        crops.append(crop_tensor)
        valid_indices.append(i)

    if not crops:
        return [{"bbox": b["bbox"], "text": "?"} for b in boxes]

    results = [None] * len(boxes)

    for start in range(0, len(crops), batch_size):
        batch = torch.stack(crops[start:start + batch_size]).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            outputs = recognizer(batch)
        _, predicted = outputs.max(1)

        for j, pred_idx in enumerate(predicted.cpu().numpy()):
            box_idx = valid_indices[start + j]
            folder_id = idx_to_class.get(int(pred_idx), str(pred_idx))
            char_text = id_to_char.get(str(folder_id), "?")
            results[box_idx] = {
                "bbox": boxes[box_idx]["bbox"],
                "text": char_text,
            }

    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"bbox": boxes[i]["bbox"], "text": "?"}

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end ancient char OCR inference")
    parser.add_argument("--input_dir", type=str,
                        default="/saisdata/13/eval/images")
    parser.add_argument("--det_model", type=str,
                        default="/app/models/best_det.pt")
    parser.add_argument("--rec_model", type=str,
                        default="/app/models/best_rec.pt")
    parser.add_argument("--char_dict", type=str,
                        default="/app/models/char_dict.json")
    parser.add_argument("--output", type=str,
                        default="/saisresult/prediction.json")
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--det_iou", type=float, default=0.45)
    parser.add_argument("--det_imgsz", type=int, default=1280)
    parser.add_argument("--rec_batch", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        for candidate in [Path("/saisdata/13/eval/images"),
                          Path("/saisdata/eval/images"),
                          Path("/saisdata")]:
            if candidate.exists() and list(candidate.glob("*.png")):
                input_dir = candidate
                break

    print(f"Input dir: {input_dir}")
    print(f"Det model: {args.det_model}")
    print(f"Rec model: {args.rec_model}")
    print(f"Char dict: {args.char_dict}")
    print(f"Device:    {args.device}")

    with open(args.char_dict, "r", encoding="utf-8") as f:
        char_dict = json.load(f)
    id_to_char = char_dict["id_to_char"]

    detector = load_detector(args.det_model, args.device)
    recognizer, rec_img_size, idx_to_class = load_recognizer(args.rec_model, args.device)
    rec_transform = build_rec_transform(rec_img_size)

    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} images to process")

    predictions = {}
    t_start = time.time()

    for i, img_path in enumerate(image_files):
        image_id = img_path.stem

        boxes = run_detection(
            detector, str(img_path),
            conf_thresh=args.det_conf,
            iou_thresh=args.det_iou,
            imgsz=args.det_imgsz,
            device=args.device,
        )

        img = Image.open(img_path).convert("RGB")

        results = run_recognition(
            recognizer, rec_transform, img, boxes,
            idx_to_class, id_to_char, args.device,
            batch_size=args.rec_batch,
        )

        img.close()

        predictions[image_id] = [{"bbox": r["bbox"], "text": r["text"]} for r in results]

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{len(image_files)}] {fps:.1f} img/s, "
                  f"{sum(len(v) for v in predictions.values())} total chars")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    total_chars = sum(len(v) for v in predictions.values())
    print(f"\nInference complete:")
    print(f"  Images:    {len(predictions)}")
    print(f"  Total chars: {total_chars}")
    print(f"  Time:      {elapsed:.1f}s ({len(predictions)/elapsed:.1f} img/s)")
    print(f"  Output:    {output_path}")


if __name__ == "__main__":
    main()
