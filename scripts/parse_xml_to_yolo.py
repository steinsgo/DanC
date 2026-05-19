#!/usr/bin/env python3
"""
parse_xml_to_yolo.py — Convert XML annotations to YOLO detection format.

For the detection stage, ALL characters share class_id=0 (single-class detection).
Character recognition is handled separately in Stage 2.

Usage:
    python scripts/parse_xml_to_yolo.py \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --output_dir /home/apulis-dev/userdata/lbh/danc/yolo_dataset \
        --val_ratio 0.15 \
        --seed 42
"""
import argparse
import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


def parse_position(position_str: str) -> Optional[tuple]:
    """
    Parse coordinate string into (x_min, y_min, x_max, y_max).
    Handles:
      - Rectangle: "x1,y1,x2,y2"
      - Polygon:   "x1,y1;x2,y2;x3,y3;..."
    """
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
            elif len(parts) >= 6 and len(parts) % 2 == 0:
                xs = [float(parts[i].strip()) for i in range(0, len(parts), 2)]
                ys = [float(parts[i].strip()) for i in range(1, len(parts), 2)]
                return (min(xs), min(ys), max(xs), max(ys))
            else:
                return None
    except (ValueError, IndexError):
        return None


def xyxy_to_yolo(x_min: float, y_min: float, x_max: float, y_max: float,
                 img_w: int, img_h: int) -> Optional[tuple]:
    """Convert (x_min, y_min, x_max, y_max) to normalized YOLO (cx, cy, w, h)."""
    if img_w <= 0 or img_h <= 0:
        return None

    bw = x_max - x_min
    bh = y_max - y_min
    if bw <= 0 or bh <= 0:
        return None

    cx = (x_min + x_max) / 2.0 / img_w
    cy = (y_min + y_max) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h

    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    nw = max(0.0, min(1.0, nw))
    nh = max(0.0, min(1.0, nh))

    return (cx, cy, nw, nh)


def parse_single_xml(xml_path: Path) -> Optional[dict]:
    """
    Parse one XML file. Returns dict with keys:
        image_id, img_w, img_h, annotations: list of (text, x_min, y_min, x_max, y_max)
    Returns None on failure.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"  [WARN] XML parse error in {xml_path.name}: {e}")
        return None

    root = tree.getroot()

    page = root if root.tag.lower() == "page" else root.find("page") or root.find("Page")
    if page is None:
        for child in root:
            if "page" in child.tag.lower():
                page = child
                break
    if page is None:
        page = root

    image_id = None
    img_w, img_h = 0, 0

    id_el = page.find("id") or page.find("ID") or page.find("Id")
    if id_el is not None and id_el.text:
        image_id = id_el.text.strip()
    if page.get("id"):
        image_id = page.get("id").strip()

    for tag in ["width", "Width"]:
        el = page.find(tag)
        if el is not None and el.text:
            try:
                img_w = int(float(el.text.strip()))
            except ValueError:
                pass
            break
    if page.get("width"):
        try:
            img_w = int(float(page.get("width")))
        except ValueError:
            pass

    for tag in ["height", "Height"]:
        el = page.find(tag)
        if el is not None and el.text:
            try:
                img_h = int(float(el.text.strip()))
            except ValueError:
                pass
            break
    if page.get("height"):
        try:
            img_h = int(float(page.get("height")))
        except ValueError:
            pass

    if image_id is None:
        image_id = xml_path.stem

    annotations = []
    char_elements = page.findall("char") or page.findall("Char")
    if not char_elements:
        char_elements = list(root.iter("char")) + list(root.iter("Char"))

    for char_el in char_elements:
        text_el = char_el.find("text") or char_el.find("Text")
        text = ""
        if text_el is not None and text_el.text:
            text = text_el.text.strip()
        elif char_el.text and char_el.text.strip():
            text = char_el.text.strip()

        pos_el = char_el.find("position") or char_el.find("Position")
        if pos_el is None or not pos_el.text:
            continue
        bbox = parse_position(pos_el.text)
        if bbox is None:
            continue

        annotations.append({
            "text": text,
            "x_min": bbox[0],
            "y_min": bbox[1],
            "x_max": bbox[2],
            "y_max": bbox[3],
        })

    return {
        "image_id": image_id,
        "img_w": img_w,
        "img_h": img_h,
        "annotations": annotations,
    }


def process_all_xmls(train_dir: Path):
    """Parse all XMLs and return list of parsed records + collected characters."""
    xml_files = sorted(train_dir.glob("*.xml"))
    if not xml_files:
        xml_files = sorted(train_dir.rglob("*.xml"))

    print(f"Found {len(xml_files)} XML files")

    records = []
    all_chars = set()
    skipped = 0
    total_annotations = 0

    for i, xml_path in enumerate(xml_files):
        rec = parse_single_xml(xml_path)
        if rec is None:
            skipped += 1
            continue

        png_candidates = [
            train_dir / f"{rec['image_id']}.png",
            train_dir / f"{rec['image_id']}",
            train_dir / f"{xml_path.stem}.png",
        ]
        png_path = None
        for c in png_candidates:
            if c.exists():
                png_path = c
                break

        if png_path is None:
            print(f"  [WARN] No matching PNG for {xml_path.name} (tried image_id={rec['image_id']})")
            skipped += 1
            continue

        if rec["img_w"] <= 0 or rec["img_h"] <= 0:
            try:
                from PIL import Image
                with Image.open(png_path) as img:
                    rec["img_w"], rec["img_h"] = img.size
            except ImportError:
                print(f"  [WARN] Cannot determine image size for {png_path.name} (Pillow not available)")
                skipped += 1
                continue
            except Exception as e:
                print(f"  [WARN] Failed to read image {png_path.name}: {e}")
                skipped += 1
                continue

        rec["png_path"] = png_path
        records.append(rec)

        for ann in rec["annotations"]:
            if ann["text"]:
                all_chars.add(ann["text"])
            total_annotations += 1

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(xml_files)} XMLs...")

    print(f"\nParsing complete:")
    print(f"  Valid images: {len(records)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total character annotations: {total_annotations}")
    print(f"  Unique characters in training data: {len(all_chars)}")

    return records, all_chars


def write_yolo_dataset(records: list, all_chars: set, output_dir: Path,
                       val_ratio: float, seed: int):
    """Write YOLO-format dataset with train/val split."""
    img_train = output_dir / "images" / "train"
    img_val = output_dir / "images" / "val"
    lbl_train = output_dir / "labels" / "train"
    lbl_val = output_dir / "labels" / "val"

    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    val_count = int(len(records) * val_ratio)
    val_indices = set(indices[:val_count])

    stats = {"train": 0, "val": 0, "train_boxes": 0, "val_boxes": 0, "empty_labels": 0}

    for idx, rec in enumerate(records):
        split = "val" if idx in val_indices else "train"
        img_dst_dir = img_val if split == "val" else img_train
        lbl_dst_dir = lbl_val if split == "val" else lbl_train

        stem = rec["png_path"].stem
        dst_img = img_dst_dir / rec["png_path"].name

        if not dst_img.exists():
            os.symlink(rec["png_path"].resolve(), dst_img)

        label_lines = []
        for ann in rec["annotations"]:
            yolo = xyxy_to_yolo(ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"],
                                rec["img_w"], rec["img_h"])
            if yolo is None:
                continue
            cx, cy, nw, nh = yolo
            label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        lbl_path = lbl_dst_dir / f"{stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
            if label_lines:
                f.write("\n")

        if not label_lines:
            stats["empty_labels"] += 1

        stats[split] += 1
        stats[f"{split}_boxes"] += len(label_lines)

    yaml_content = (
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: 1\n"
        f"names: ['char']\n"
    )

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    char_list_path = output_dir / "train_chars.txt"
    sorted_chars = sorted(all_chars)
    with open(char_list_path, "w", encoding="utf-8") as f:
        for ch in sorted_chars:
            f.write(ch + "\n")

    print(f"\nDataset written to: {output_dir}")
    print(f"  Train images: {stats['train']} ({stats['train_boxes']} boxes)")
    print(f"  Val images:   {stats['val']} ({stats['val_boxes']} boxes)")
    print(f"  Empty labels: {stats['empty_labels']}")
    print(f"  dataset.yaml: {yaml_path}")
    print(f"  train_chars.txt: {char_list_path} ({len(sorted_chars)} unique chars)")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Convert XML annotations to YOLO format")
    parser.add_argument("--train_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/train/out_of_domain",
                        help="Directory containing PNG + XML training files")
    parser.add_argument("--output_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/yolo_dataset",
                        help="Output directory for YOLO dataset")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of data for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible split")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)

    if not train_dir.exists():
        print(f"ERROR: Training directory not found: {train_dir}")
        return

    records, all_chars = process_all_xmls(train_dir)

    if not records:
        print("ERROR: No valid records parsed. Run explore_data.py first to check format.")
        return

    yaml_path = write_yolo_dataset(records, all_chars, output_dir, args.val_ratio, args.seed)
    print(f"\nPhase 1 (Detection Data) complete. Use this for YOLO training:")
    print(f"  dataset.yaml = {yaml_path}")


if __name__ == "__main__":
    main()
