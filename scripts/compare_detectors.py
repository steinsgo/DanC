#!/usr/bin/env python3
"""
compare_detectors.py — 对比新旧YOLO检测权重在同一批训练图上的检测效果。

使用XML标注作为ground truth，统计召回率/准确率（IoU>=0.5）。

用法:
    python scripts/compare_detectors.py \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --old_model /home/apulis-dev/userdata/lbh/danc/DanC/models/best_det.pt \
        --new_model /home/apulis-dev/userdata/lbh/danc/runs/detect/det_v2_1536/weights/best.pt \
        --num_images 200 \
        --imgsz_old 1280 \
        --imgsz_new 1536
"""
import argparse
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def read_xml_text(xml_path: Path) -> str:
    raw = xml_path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode("utf-16")
    if raw[:3] == b'\xef\xbb\xbf':
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-16")


def parse_xml_auto(xml_path: Path):
    text = read_xml_text(xml_path)
    if text and text[0] == '﻿':
        text = text[1:]
    return ET.fromstring(text)


def parse_position(pos_str: str) -> Optional[tuple]:
    pos_str = pos_str.strip()
    if not pos_str:
        return None
    try:
        if ";" in pos_str:
            xs, ys = [], []
            for pt in pos_str.split(";"):
                pt = pt.strip()
                if not pt:
                    continue
                c = pt.split(",")
                xs.append(float(c[0])); ys.append(float(c[1]))
            if len(xs) < 3:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        parts = pos_str.split(",")
        if len(parts) == 4:
            x1, y1, x2, y2 = [float(p) for p in parts]
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return None
    except (ValueError, IndexError):
        return None


def get_gt_boxes(xml_path: Path):
    """Returns list of (x1, y1, x2, y2) ground truth boxes."""
    try:
        root = parse_xml_auto(xml_path)
    except Exception:
        return []
    boxes = []
    for char_el in root.iter("char"):
        pos = char_el.get("position", "")
        bbox = parse_position(pos)
        if bbox is not None:
            boxes.append(bbox)
    return boxes


def iou_xyxy(a, b):
    """IoU of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_detector(model, image_paths, gt_dict, conf, iou_thresh,
                      imgsz, device, iou_match=0.5):
    """Run detector on images, compute matched/total stats."""
    total_gt = 0
    total_pred = 0
    matched = 0

    for img_path in image_paths:
        gt_boxes = gt_dict.get(img_path.stem, [])
        total_gt += len(gt_boxes)

        results = model.predict(
            source=str(img_path), conf=conf, iou=iou_thresh,
            imgsz=imgsz, verbose=False, device=device,
        )
        pred_boxes = []
        if results and len(results) > 0 and results[0].boxes is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            for row in xyxy:
                pred_boxes.append((row[0], row[1], row[2], row[3]))
        total_pred += len(pred_boxes)

        # Greedy IoU matching: each GT matched at most once
        used_pred = [False] * len(pred_boxes)
        for gt in gt_boxes:
            best_iou = 0.0
            best_j = -1
            for j, pb in enumerate(pred_boxes):
                if used_pred[j]:
                    continue
                v = iou_xyxy(gt, pb)
                if v > best_iou:
                    best_iou = v
                    best_j = j
            if best_iou >= iou_match and best_j >= 0:
                used_pred[best_j] = True
                matched += 1

    return {
        "gt": total_gt,
        "pred": total_pred,
        "matched": matched,
        "recall": matched / total_gt if total_gt > 0 else 0.0,
        "precision": matched / total_pred if total_pred > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare detector models")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--old_model", type=str, required=True)
    parser.add_argument("--new_model", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--iou_thresh", type=float, default=0.45)
    parser.add_argument("--iou_match", type=float, default=0.5,
                        help="IoU threshold to count as a match (eval metric)")
    parser.add_argument("--imgsz_old", type=int, default=1280)
    parser.add_argument("--imgsz_new", type=int, default=1536)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)

    # Sample images that have both PNG and XML
    xml_files = sorted(train_dir.glob("*.xml"))
    random.seed(args.seed)
    random.shuffle(xml_files)

    selected_pngs = []
    gt_dict = {}
    for xml_path in xml_files:
        png_path = train_dir / f"{xml_path.stem}.png"
        if not png_path.exists():
            continue
        gt = get_gt_boxes(xml_path)
        if not gt:
            continue
        gt_dict[png_path.stem] = gt
        selected_pngs.append(png_path)
        if len(selected_pngs) >= args.num_images:
            break

    print(f"Selected {len(selected_pngs)} images, "
          f"{sum(len(v) for v in gt_dict.values())} GT boxes total")
    print(f"IoU match threshold: {args.iou_match}")
    print(f"Conf threshold:      {args.conf}\n")

    from ultralytics import YOLO

    print(f"=== OLD model: {args.old_model} (imgsz={args.imgsz_old}) ===")
    old_model = YOLO(args.old_model)
    old_stats = evaluate_detector(
        old_model, selected_pngs, gt_dict, args.conf, args.iou_thresh,
        args.imgsz_old, args.device, args.iou_match,
    )
    print(f"  GT: {old_stats['gt']}  Pred: {old_stats['pred']}  Matched: {old_stats['matched']}")
    print(f"  Recall:    {old_stats['recall']:.4f}")
    print(f"  Precision: {old_stats['precision']:.4f}\n")

    print(f"=== NEW model: {args.new_model} (imgsz={args.imgsz_new}) ===")
    new_model = YOLO(args.new_model)
    new_stats = evaluate_detector(
        new_model, selected_pngs, gt_dict, args.conf, args.iou_thresh,
        args.imgsz_new, args.device, args.iou_match,
    )
    print(f"  GT: {new_stats['gt']}  Pred: {new_stats['pred']}  Matched: {new_stats['matched']}")
    print(f"  Recall:    {new_stats['recall']:.4f}")
    print(f"  Precision: {new_stats['precision']:.4f}\n")

    delta_r = new_stats['recall'] - old_stats['recall']
    delta_p = new_stats['precision'] - old_stats['precision']
    print(f"=== Delta (new - old) ===")
    print(f"  Recall:    {delta_r:+.4f}")
    print(f"  Precision: {delta_p:+.4f}")
    if delta_r > 0.02:
        print(f"\n>> Recommendation: NEW model wins, ship it.")
    elif delta_r < -0.02:
        print(f"\n>> Recommendation: NEW model regressed, keep OLD.")
    else:
        print(f"\n>> Recommendation: similar; keep current submission, save the slot.")


if __name__ == "__main__":
    main()
