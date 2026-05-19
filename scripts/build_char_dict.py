#!/usr/bin/env python3
"""
build_char_dict.py — Build global character dictionary from HUST-OBC + training XML.

Scans all HUST-OBC annotation JSONs and training XML char/text fields, assigns
each unique character a stable integer class_id for the Stage 2 recognition model.

Usage:
    python scripts/build_char_dict.py \
        --hust_dir  /home/apulis-dev/userdata/lbh/danc/HUST-OBC \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --output    /home/apulis-dev/userdata/lbh/danc/char_dict.json
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def scan_hust_obc(hust_dir: Path) -> dict:
    """
    Walk HUST-OBC directory tree and extract character metadata.
    Returns: {char_text: {"unicode": str, "count": int, "source": "hust-obc"}}

    Handles multiple possible structures:
      1. hust_dir/deciphered/XXXX/ containing JSON + images
      2. hust_dir/deciphered/XXXX/YYYY/ nested subdirectories
    """
    char_info = {}
    json_count = 0
    image_count = 0

    for category in ["deciphered", "undeciphered"]:
        cat_dir = hust_dir / category
        if not cat_dir.exists():
            continue

        for json_path in sorted(cat_dir.rglob("*.json")):
            json_count += 1
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"  [WARN] Failed to read {json_path}: {e}")
                continue

            entries = data if isinstance(data, list) else [data]

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                char_text = entry.get("char", "") or entry.get("text", "") or ""
                char_unicode = entry.get("char_unicode", "") or entry.get("unicode", "") or ""

                if not char_text and not char_unicode:
                    continue

                key = char_text if char_text else f"U+{char_unicode}"
                if key not in char_info:
                    char_info[key] = {
                        "char": char_text,
                        "unicode": char_unicode,
                        "count": 0,
                        "source": "hust-obc",
                        "category": category,
                    }
                char_info[key]["count"] += 1

        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
            image_count += len(list(cat_dir.rglob(ext)))

    print(f"  HUST-OBC scan: {json_count} JSON files, ~{image_count} images")
    print(f"  Unique characters from HUST-OBC: {len(char_info)}")

    return char_info


def scan_hust_obc_by_dirname(hust_dir: Path) -> dict:
    """
    Fallback: if JSON files lack char labels, use directory names as character IDs.
    Each leaf directory = one character class; images inside = instances.
    """
    char_info = {}

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
                leaf_dirs.append(Path(dirpath))

        for leaf in leaf_dirs:
            n_imgs = sum(
                1 for f in leaf.iterdir()
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            rel_path = leaf.relative_to(hust_dir)
            dir_id = str(rel_path)

            json_files = list(leaf.glob("*.json"))
            char_text = ""
            char_unicode = ""

            for jf in json_files:
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    entries = data if isinstance(data, list) else [data]
                    for entry in entries:
                        if isinstance(entry, dict):
                            char_text = entry.get("char", "") or entry.get("text", "") or char_text
                            char_unicode = entry.get("char_unicode", "") or entry.get("unicode", "") or char_unicode
                            if char_text:
                                break
                    if char_text:
                        break
                except Exception:
                    continue

            key = char_text if char_text else dir_id
            char_info[key] = {
                "char": char_text,
                "unicode": char_unicode,
                "dir_id": dir_id,
                "count": n_imgs,
                "source": "hust-obc",
                "category": category,
            }

    print(f"  HUST-OBC dir-based scan: {len(char_info)} character classes")
    return char_info


def scan_train_xmls(train_dir: Path) -> Counter:
    """Extract all character texts from training XMLs."""
    char_counter = Counter()

    xml_files = sorted(train_dir.glob("*.xml"))
    if not xml_files:
        xml_files = sorted(train_dir.rglob("*.xml"))

    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        for char_el in list(root.iter("char")) + list(root.iter("Char")):
            text_el = char_el.find("text") or char_el.find("Text")
            text = ""
            if text_el is not None and text_el.text:
                text = text_el.text.strip()
            elif char_el.text and char_el.text.strip():
                text = char_el.text.strip()

            if text:
                char_counter[text] += 1

    print(f"  Training XML scan: {len(char_counter)} unique characters, "
          f"{sum(char_counter.values())} total instances")
    return char_counter


def build_dictionary(hust_chars: dict, train_chars: Counter, output_path: Path):
    """
    Merge HUST-OBC and training characters into a unified dictionary.
    Assigns stable integer class IDs sorted by: training frequency desc, then HUST-OBC frequency desc.
    """
    all_chars = {}

    for key, info in hust_chars.items():
        char_text = info.get("char", key)
        if not char_text:
            char_text = key
        all_chars[char_text] = {
            "hust_count": info.get("count", 0),
            "train_count": train_chars.get(char_text, 0),
            "unicode": info.get("unicode", ""),
            "dir_id": info.get("dir_id", ""),
            "category": info.get("category", ""),
        }

    for char_text, count in train_chars.items():
        if char_text not in all_chars:
            all_chars[char_text] = {
                "hust_count": 0,
                "train_count": count,
                "unicode": "",
                "dir_id": "",
                "category": "",
            }
        else:
            all_chars[char_text]["train_count"] = count

    sorted_chars = sorted(
        all_chars.keys(),
        key=lambda c: (all_chars[c]["train_count"], all_chars[c]["hust_count"]),
        reverse=True,
    )

    char_dict = {}
    for class_id, char_text in enumerate(sorted_chars):
        info = all_chars[char_text]
        char_dict[char_text] = {
            "class_id": class_id,
            "hust_count": info["hust_count"],
            "train_count": info["train_count"],
            "unicode": info["unicode"],
            "dir_id": info["dir_id"],
        }

    id_to_char = {v["class_id"]: k for k, v in char_dict.items()}

    output = {
        "num_classes": len(char_dict),
        "char_to_id": {k: v["class_id"] for k, v in char_dict.items()},
        "id_to_char": {str(k): v for k, v in id_to_char.items()},
        "details": char_dict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    in_both = sum(1 for c in all_chars if all_chars[c]["train_count"] > 0 and all_chars[c]["hust_count"] > 0)
    train_only = sum(1 for c in all_chars if all_chars[c]["train_count"] > 0 and all_chars[c]["hust_count"] == 0)
    hust_only = sum(1 for c in all_chars if all_chars[c]["hust_count"] > 0 and all_chars[c]["train_count"] == 0)

    print(f"\nDictionary built: {output_path}")
    print(f"  Total classes: {len(char_dict)}")
    print(f"  In both sources: {in_both}")
    print(f"  Training-only chars: {train_only}")
    print(f"  HUST-OBC-only chars: {hust_only}")
    print(f"  Top-10 by training frequency:")
    for ch in sorted_chars[:10]:
        info = char_dict[ch]
        print(f"    [{info['class_id']}] '{ch}' train={info['train_count']} hust={info['hust_count']}")


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
        if len(hust_chars) < 100:
            print("  Few chars from JSON scan, trying directory-based scan...")
            hust_chars_dir = scan_hust_obc_by_dirname(hust_dir)
            if len(hust_chars_dir) > len(hust_chars):
                hust_chars = hust_chars_dir
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
