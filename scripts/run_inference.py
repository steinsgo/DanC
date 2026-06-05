#!/usr/bin/env python3
"""
run_inference.py — End-to-end inference pipeline for ancient character OCR.

Stage 1: YOLO detector finds character bounding boxes (multi-scale TTA optional).
Stage 2: ResNet classifier recognizes each cropped character (scale-jitter TTA optional).

Output: prediction.json

Usage (Docker default paths):
    python /app/src/run_inference.py

Usage (local):
    python scripts/run_inference.py \
        --input_dir /saisdata/13/eval/images \
        --det_model /path/to/best_det.pt \
        --rec_model /path/to/best_rec.pt \
        --char_dict /path/to/char_dict.json \
        --output /path/to/prediction.json \
        --det_imgszs 1280,1536 \
        --det_conf 0.15 \
        --rec_tta 1 \
        --crop_pad 0.1
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    state = ckpt["model_state_dict"]
    # Older checkpoints saved fc as a plain Linear (fc.weight/fc.bias).
    # Remap to Sequential(Dropout, Linear) keys if needed.
    if "fc.weight" in state and "fc.1.weight" not in state:
        state["fc.1.weight"] = state.pop("fc.weight")
        state["fc.1.bias"] = state.pop("fc.bias")
    model.load_state_dict(state)
    model.to(device).eval()

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


# ---------------------------------------------------------------------------
# Detection (multi-scale TTA + NMS merge)
# ---------------------------------------------------------------------------

def _nms_xyxy(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """Pure-numpy NMS. Returns indices to keep."""
    if len(boxes_xyxy) == 0:
        return []
    x1 = boxes_xyxy[:, 0]; y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]; y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(min=0) * (yy2 - yy1).clip(min=0)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou < iou_thresh]
    return keep


def run_detection(detector, img_path: str, conf_thresh: float, iou_thresh: float,
                  imgszs, device: str):
    """
    Run detector at one or more scales, merge with NMS.

    imgszs: list of int (e.g. [1280] or [1280, 1536]).
    Returns list of {"bbox": [x, y, w, h], "conf": float}.
    """
    all_xyxy = []
    all_scores = []
    for sz in imgszs:
        results = detector.predict(
            source=img_path, conf=conf_thresh, iou=iou_thresh,
            imgsz=sz, verbose=False, device=device,
        )
        if not results or len(results) == 0:
            continue
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        all_xyxy.append(xyxy)
        all_scores.append(confs)

    if not all_xyxy:
        return []

    xyxy = np.concatenate(all_xyxy, axis=0)
    scores = np.concatenate(all_scores, axis=0)

    if len(imgszs) > 1:
        keep = _nms_xyxy(xyxy, scores, iou_thresh)
        xyxy = xyxy[keep]; scores = scores[keep]

    boxes = []
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        boxes.append({
            "bbox": [int(round(x1)), int(round(y1)),
                     int(round(x2 - x1)), int(round(y2 - y1))],
            "conf": float(scores[i]),
        })
    return boxes


# ---------------------------------------------------------------------------
# Recognition (with optional padding + scale-jitter TTA)
# ---------------------------------------------------------------------------

def _pad_crop_box(x, y, w, h, img_w, img_h, pad_ratio: float):
    """Apply pad_ratio expansion on bbox, clamp to image. Returns (x1, y1, x2, y2)."""
    pad_x = w * pad_ratio
    pad_y = h * pad_ratio
    x1 = max(0, int(round(x - pad_x)))
    y1 = max(0, int(round(y - pad_y)))
    x2 = min(img_w, int(round(x + w + pad_x)))
    y2 = min(img_h, int(round(y + h + pad_y)))
    return x1, y1, x2, y2


def _build_tta_transforms(img_size: int):
    """Build per-scale transforms for recognition TTA (centered scale jitter)."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    base = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])
    # Two extra slightly-padded views to simulate scale jitter
    pad_small = transforms.Compose([
        transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])
    pad_large = transforms.Compose([
        transforms.Resize((int(img_size * 0.95), int(img_size * 0.95))),
        transforms.Pad(padding=(img_size - int(img_size * 0.95)) // 2 + 1, fill=255),
        transforms.CenterCrop(img_size),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm,
    ])
    return [base, pad_small, pad_large]


def run_recognition(recognizer, rec_transform, tta_transforms, img: Image.Image,
                    boxes: list, idx_to_class: dict, id_to_char: dict,
                    device: str, batch_size: int, crop_pad: float, use_tta: bool):
    if not boxes:
        return []

    img_w, img_h = img.size
    crops_per_view = [[] for _ in range(len(tta_transforms) if use_tta else 1)]
    valid_indices = []

    for i, box_info in enumerate(boxes):
        x, y, w, h = box_info["bbox"]
        x1, y1, x2, y2 = _pad_crop_box(x, y, w, h, img_w, img_h, crop_pad)
        if x2 - x1 <= 2 or y2 - y1 <= 2:
            continue
        crop = img.crop((x1, y1, x2, y2))

        if use_tta:
            for v, tf in enumerate(tta_transforms):
                crops_per_view[v].append(tf(crop))
        else:
            crops_per_view[0].append(rec_transform(crop))
        valid_indices.append(i)

    if not valid_indices:
        return [{"bbox": b["bbox"], "text": "?"} for b in boxes]

    results = [None] * len(boxes)

    n = len(valid_indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        # Average softmax over views
        prob_sum = None
        for view_crops in crops_per_view:
            batch = torch.stack(view_crops[start:end]).to(device)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                logits = recognizer(batch)
            probs = F.softmax(logits.float(), dim=1)
            prob_sum = probs if prob_sum is None else prob_sum + probs

        avg_probs = prob_sum / len(crops_per_view)
        _, predicted = avg_probs.max(1)

        for j, pred_idx in enumerate(predicted.cpu().numpy()):
            box_idx = valid_indices[start + j]
            folder_id = idx_to_class.get(int(pred_idx), str(pred_idx))
            char_text = id_to_char.get(str(folder_id), "?")
            results[box_idx] = {"bbox": boxes[box_idx]["bbox"], "text": char_text}

    for i in range(len(results)):
        if results[i] is None:
            results[i] = {"bbox": boxes[i]["bbox"], "text": "?"}
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end ancient char OCR inference")
    parser.add_argument("--input_dir", type=str, default="/saisdata/13/eval/images")
    parser.add_argument("--det_model", type=str, default="/app/models/best_det.pt")
    parser.add_argument("--rec_model", type=str, default="/app/models/best_rec.pt")
    parser.add_argument("--char_dict", type=str, default="/app/models/char_dict.json")
    parser.add_argument("--output", type=str, default="/saisresult/prediction.json")
    parser.add_argument("--det_conf", type=float, default=0.15,
                        help="Detection confidence threshold (lower = higher recall)")
    parser.add_argument("--det_iou", type=float, default=0.45)
    parser.add_argument("--det_imgszs", type=str, default="1280,1536",
                        help="Comma-separated detection scales for multi-scale TTA")
    parser.add_argument("--rec_batch", type=int, default=128)
    parser.add_argument("--rec_tta", type=int, default=1,
                        help="1 = enable scale-jitter recognition TTA, 0 = disable")
    parser.add_argument("--crop_pad", type=float, default=0.1,
                        help="Padding ratio applied to detected bbox before recognition (matches train)")
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

    det_imgszs = [int(s) for s in args.det_imgszs.split(",") if s.strip()]
    use_tta = bool(args.rec_tta)

    print(f"Input dir:   {input_dir}")
    print(f"Det model:   {args.det_model}")
    print(f"Rec model:   {args.rec_model}")
    print(f"Char dict:   {args.char_dict}")
    print(f"Device:      {args.device}")
    print(f"Det scales:  {det_imgszs}")
    print(f"Det conf:    {args.det_conf}")
    print(f"Crop pad:    {args.crop_pad}")
    print(f"Rec TTA:     {'on' if use_tta else 'off'}")

    with open(args.char_dict, "r", encoding="utf-8") as f:
        char_dict = json.load(f)
    id_to_char = char_dict["id_to_char"]

    detector = load_detector(args.det_model, args.device)
    recognizer, rec_img_size, idx_to_class = load_recognizer(args.rec_model, args.device)
    rec_transform = build_rec_transform(rec_img_size)
    tta_transforms = _build_tta_transforms(rec_img_size) if use_tta else [rec_transform]

    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))
    print(f"Found {len(image_files)} images to process")

    predictions = {}
    t_start = time.time()

    for i, img_path in enumerate(image_files):
        image_id = img_path.stem

        boxes = run_detection(
            detector, str(img_path),
            conf_thresh=args.det_conf, iou_thresh=args.det_iou,
            imgszs=det_imgszs, device=args.device,
        )

        img = Image.open(img_path).convert("RGB")
        results = run_recognition(
            recognizer, rec_transform, tta_transforms, img, boxes,
            idx_to_class, id_to_char, args.device,
            batch_size=args.rec_batch, crop_pad=args.crop_pad, use_tta=use_tta,
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
    print(f"  Images:      {len(predictions)}")
    print(f"  Total chars: {total_chars}")
    print(f"  Time:        {elapsed:.1f}s ({len(predictions)/elapsed:.1f} img/s)")
    print(f"  Output:      {output_path}")


if __name__ == "__main__":
    main()
