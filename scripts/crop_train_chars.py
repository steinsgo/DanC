#!/usr/bin/env python3
"""
crop_train_chars.py — Crop individual characters from training images using XML annotations.
Creates a classification dataset: output_dir/{char_text}/image_stem_charN.png

This provides labeled training data for the Stage 2 recognizer.

Usage:
    python scripts/crop_train_chars.py \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --output_dir /home/apulis-dev/userdata/lbh/danc/cropped_chars \
        --char_dict /home/apulis-dev/userdata/lbh/danc/char_dict.json \
        --val_ratio 0.15 --seed 42
"""
import argparse
import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


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


def parse_xml_auto(xml_path: Path) -> ET.Element:
    text = read_xml_text(xml_path)
    if text and text[0] == '﻿':
        text = text[1:]
    return ET.fromstring(text)


def parse_position(position_str: str):
    position_str = position_str.strip()
    if not position_str:
        return None
    try:
        if ";" in position_str:
            points = position_str.split(";")
            xs, ys = [], []
            for pt in points:
                pt = pt.strip()
                if not pt:
                    continue
                coords = pt.split(",")
                xs.append(float(coords[0].strip()))
                ys.append(float(coords[1].strip()))
            if len(xs) < 3:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        else:
            parts = position_str.split(",")
            if len(parts) == 4:
                x1, y1, x2, y2 = [float(p.strip()) for p in parts]
                return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return None
    except (ValueError, IndexError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Crop characters from training images")
    parser.add_argument("--train_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/train/out_of_domain")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/cropped_chars")
    parser.add_argument("--char_dict", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/char_dict.json")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pad_ratio", type=float, default=0.1,
                        help="Padding around crop as fraction of bbox size")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)

    with open(args.char_dict, "r", encoding="utf-8") as f:
        char_dict = json.load(f)
    char_to_id = char_dict["char_to_id"]
    print(f"Loaded char_dict: {len(char_to_id)} classes")

    xml_files = sorted(train_dir.glob("*.xml"))
    print(f"Found {len(xml_files)} XML files")

    random.seed(args.seed)
    random.shuffle(xml_files)
    val_count = int(len(xml_files) * args.val_ratio)
    val_xmls = set(x.name for x in xml_files[:val_count])

    stats = {"train": 0, "val": 0, "skipped_nochar": 0, "skipped_bbox": 0}

    for i, xml_path in enumerate(sorted(train_dir.glob("*.xml"))):
        try:
            root = parse_xml_auto(xml_path)
        except (ET.ParseError, UnicodeDecodeError):
            continue

        page = root if root.tag.lower() == "page" else root
        raw_id = page.get("id", "")
        stem = Path(raw_id).stem if raw_id else xml_path.stem

        png_path = train_dir / f"{stem}.png"
        if not png_path.exists():
            png_path = train_dir / f"{xml_path.stem}.png"
        if not png_path.exists():
            continue

        split = "val" if xml_path.name in val_xmls else "train"

        try:
            img = Image.open(png_path)
        except Exception:
            continue

        img_w, img_h = img.size
        char_idx = 0

        for char_el in root.iter("char"):
            char_text = (char_el.text or "").strip()
            if not char_text or char_text not in char_to_id:
                stats["skipped_nochar"] += 1
                continue

            pos_str = char_el.get("position", "")
            if not pos_str:
                continue

            bbox = parse_position(pos_str)
            if bbox is None:
                stats["skipped_bbox"] += 1
                continue

            x_min, y_min, x_max, y_max = bbox

            # Clamp bbox to image bounds first
            x_min = max(0, min(x_min, img_w - 1))
            y_min = max(0, min(y_min, img_h - 1))
            x_max = max(0, min(x_max, img_w))
            y_max = max(0, min(y_max, img_h))

            # Recompute width/height after clamping
            bw = x_max - x_min
            bh = y_max - y_min
            if bw <= 2 or bh <= 2:
                stats["skipped_bbox"] += 1
                continue

            pad_x = bw * args.pad_ratio
            pad_y = bh * args.pad_ratio
            crop_x1 = max(0, int(x_min - pad_x))
            crop_y1 = max(0, int(y_min - pad_y))
            crop_x2 = min(img_w, int(x_max + pad_x))
            crop_y2 = min(img_h, int(y_max + pad_y))

            # Final safety check: skip if crop area is invalid
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                stats["skipped_bbox"] += 1
                continue

            crop = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            class_id = char_to_id[char_text]
            class_dir = output_dir / split / str(class_id)
            class_dir.mkdir(parents=True, exist_ok=True)

            crop_name = f"{stem}_{char_idx}.png"
            crop.save(class_dir / crop_name)

            stats[split] += 1
            char_idx += 1

        img.close()

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1} images... (train={stats['train']}, val={stats['val']})")

    meta = {
        "num_classes": len(char_to_id),
        "char_to_id": char_to_id,
        "id_to_char": char_dict["id_to_char"],
        "train_samples": stats["train"],
        "val_samples": stats["val"],
    }
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nCropping complete:")
    print(f"  Train crops: {stats['train']}")
    print(f"  Val crops:   {stats['val']}")
    print(f"  Skipped (no char in dict): {stats['skipped_nochar']}")
    print(f"  Skipped (bad bbox): {stats['skipped_bbox']}")
    print(f"  Output: {output_dir}")
    print(f"  Meta:   {meta_path}")


if __name__ == "__main__":
    main()
