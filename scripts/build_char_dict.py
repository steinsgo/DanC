#!/usr/bin/env python3
"""
build_char_dict.py — Build global character dictionary from HUST-OBC + training XML.

HUST-OBC structure (verified from server):
    deciphered/0001/  -> 35 PNG images, NO JSON
    deciphered/0002/  -> 53 PNG images, NO JSON
    undeciphered/L/1/ -> 1 JPG image
    ...
    Filenames encode source: H_0001_60BB6_0.png, G_0001_#Uxxxx...png
    Also: Source.txt, GuoXueDaShi_1390/ at top level

Training XML structure (verified):
    <char id="1" position="x1,y1,x2,y2" ...>牛</char>
    Character text is inline, position is attribute.

Usage:
    python scripts/build_char_dict.py \
        --hust_dir  /home/apulis-dev/userdata/lbh/danc/HUST-OBC \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --output    /home/apulis-dev/userdata/lbh/danc/char_dict.json
"""
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def read_xml_text(xml_path: Path) -> str:
    """Read XML file with automatic encoding detection (UTF-8 / UTF-16)."""
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
    """Parse XML with automatic encoding handling."""
    text = read_xml_text(xml_path)
    if text and text[0] == '﻿':
        text = text[1:]
    return ET.fromstring(text)


def scan_hust_obc(hust_dir: Path) -> dict:
    """
    Walk HUST-OBC directory tree. No JSON files exist — character class is
    identified by directory name. For deciphered chars, we try to read
    Source.txt or GuoXueDaShi mapping if available.

    Returns: {dir_id: {"count": int, "category": str, ...}}
    """
    char_info = {}

    source_map = _load_source_txt(hust_dir)
    if source_map:
        print(f"  Loaded Source.txt with {len(source_map)} mappings")

    total_images = 0

    for category in ["deciphered", "undeciphered"]:
        cat_dir = hust_dir / category
        if not cat_dir.exists():
            continue

        leaf_dirs = []
        for dirpath, dirnames, filenames in os.walk(cat_dir):
            has_images = any(
                f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                for f in filenames
            )
            if has_images:
                leaf_dirs.append((Path(dirpath), filenames))

        for leaf, filenames in leaf_dirs:
            img_files = [
                f for f in filenames
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
            n_imgs = len(img_files)
            total_images += n_imgs

            rel_path = leaf.relative_to(hust_dir)
            dir_id = str(rel_path)

            dir_name = leaf.name

            char_text = source_map.get(dir_name, "")

            if not char_text and category == "deciphered" and img_files:
                char_text = _extract_char_from_filename(img_files[0])

            char_info[dir_id] = {
                "char": char_text,
                "dir_id": dir_id,
                "dir_name": dir_name,
                "count": n_imgs,
                "source": "hust-obc",
                "category": category,
            }

    print(f"  HUST-OBC scan: {len(char_info)} character classes, {total_images} total images")
    deciphered_with_char = sum(
        1 for v in char_info.values()
        if v["category"] == "deciphered" and v["char"]
    )
    print(f"  Deciphered classes with known char: {deciphered_with_char}")

    return char_info


def _load_source_txt(hust_dir: Path) -> dict:
    """Try to load Source.txt as a dir_name -> character mapping."""
    source_path = hust_dir / "Source.txt"
    if not source_path.exists():
        return {}

    mapping = {}
    try:
        raw = source_path.read_bytes()
        if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
            text = raw.decode("utf-16")
        elif raw[:3] == b'\xef\xbb\xbf':
            text = raw.decode("utf-8-sig")
        else:
            text = raw.decode("utf-8", errors="replace")

        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 2:
                dir_name = parts[0].strip()
                char_text = parts[1].strip()
                if char_text:
                    mapping[dir_name] = char_text
            elif len(parts) == 1 and "\t" in line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    mapping[parts[0].strip()] = parts[1].strip()
    except Exception as e:
        print(f"  [WARN] Failed to parse Source.txt: {e}")

    return mapping


def _extract_char_from_filename(filename: str) -> str:
    """
    Try to extract character from HUST-OBC filename patterns.
    This is a best-effort heuristic; may not always work.
    """
    return ""


def scan_train_xmls(train_dir: Path) -> Counter:
    """Extract all character texts from training XMLs (inline text in <char> elements)."""
    char_counter = Counter()

    xml_files = sorted(train_dir.glob("*.xml"))
    if not xml_files:
        xml_files = sorted(train_dir.rglob("*.xml"))

    for xml_path in xml_files:
        try:
            root = parse_xml_auto(xml_path)
        except (ET.ParseError, UnicodeDecodeError):
            continue

        for char_el in root.iter("char"):
            text = (char_el.text or "").strip()
            if text:
                char_counter[text] += 1

    print(f"  Training XML scan: {len(char_counter)} unique characters, "
          f"{sum(char_counter.values())} total instances")
    return char_counter


def build_dictionary(hust_chars: dict, train_chars: Counter, output_path: Path):
    """
    Merge HUST-OBC and training characters into a unified dictionary.
    """
    all_chars = {}

    for dir_id, info in hust_chars.items():
        char_text = info.get("char", "")
        if char_text and char_text not in all_chars:
            all_chars[char_text] = {
                "hust_count": info["count"],
                "train_count": train_chars.get(char_text, 0),
                "hust_dir": dir_id,
                "category": info.get("category", ""),
            }

    for char_text, count in train_chars.items():
        if char_text not in all_chars:
            all_chars[char_text] = {
                "hust_count": 0,
                "train_count": count,
                "hust_dir": "",
                "category": "",
            }
        else:
            all_chars[char_text]["train_count"] = count

    sorted_chars = sorted(
        all_chars.keys(),
        key=lambda c: (all_chars[c]["train_count"], all_chars[c]["hust_count"]),
        reverse=True,
    )

    char_to_id = {}
    for class_id, char_text in enumerate(sorted_chars):
        char_to_id[char_text] = class_id

    id_to_char = {str(v): k for k, v in char_to_id.items()}

    hust_dir_to_id = {}
    for dir_id, info in hust_chars.items():
        char_text = info.get("char", "")
        if char_text and char_text in char_to_id:
            hust_dir_to_id[dir_id] = char_to_id[char_text]

    output = {
        "num_classes": len(char_to_id),
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "hust_dir_to_class_id": hust_dir_to_id,
        "hust_stats": {
            "total_classes": len(hust_chars),
            "mapped_classes": len(hust_dir_to_id),
            "unmapped_classes": len(hust_chars) - len(hust_dir_to_id),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    in_both = sum(1 for c in all_chars if all_chars[c]["train_count"] > 0 and all_chars[c]["hust_count"] > 0)
    train_only = sum(1 for c in all_chars if all_chars[c]["train_count"] > 0 and all_chars[c]["hust_count"] == 0)
    hust_only = sum(1 for c in all_chars if all_chars[c]["hust_count"] > 0 and all_chars[c]["train_count"] == 0)

    print(f"\nDictionary built: {output_path}")
    print(f"  Total classes: {len(char_to_id)}")
    print(f"  In both sources: {in_both}")
    print(f"  Training-only chars: {train_only}")
    print(f"  HUST-OBC-only chars: {hust_only}")
    print(f"  HUST dirs mapped to dict: {len(hust_dir_to_id)}/{len(hust_chars)}")
    print(f"  Top-10 by training frequency:")
    for ch in sorted_chars[:10]:
        cid = char_to_id[ch]
        info = all_chars[ch]
        print(f"    [{cid}] '{ch}' train={info['train_count']} hust={info['hust_count']}")


def main():
    parser = argparse.ArgumentParser(description="Build global character dictionary")
    parser.add_argument("--hust_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/HUST-OBC")
    parser.add_argument("--train_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/train/out_of_domain")
    parser.add_argument("--output", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/char_dict.json")
    args = parser.parse_args()

    hust_dir = Path(args.hust_dir)
    train_dir = Path(args.train_dir)
    output_path = Path(args.output)

    print("[1/3] Scanning HUST-OBC...")
    hust_chars = {}
    if hust_dir.exists():
        hust_chars = scan_hust_obc(hust_dir)
    else:
        print(f"  WARNING: HUST-OBC dir not found: {hust_dir}")

    print("\n[2/3] Scanning training XMLs...")
    train_chars = Counter()
    if train_dir.exists():
        train_chars = scan_train_xmls(train_dir)
    else:
        print(f"  WARNING: Training dir not found: {train_dir}")

    print("\n[3/3] Building unified dictionary...")
    build_dictionary(hust_chars, train_chars, output_path)


if __name__ == "__main__":
    main()
