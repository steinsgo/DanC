#!/usr/bin/env python3
"""
explore_data.py — Run FIRST on server to verify data formats before conversion.

Usage:
    python scripts/explore_data.py \
        --train_dir /home/apulis-dev/userdata/lbh/danc/train/out_of_domain \
        --hust_dir  /home/apulis-dev/userdata/lbh/danc/HUST-OBC
"""
import argparse
import json
import os
import xml.etree.ElementTree as ET
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


def explore_train_xml(train_dir: Path, max_files: int = 3):
    print("=" * 70)
    print(f"[1] TRAINING DATA: {train_dir}")
    print("=" * 70)

    if not train_dir.exists():
        print("  ERROR: directory does not exist!")
        return

    pngs = sorted(train_dir.glob("*.png"))
    xmls = sorted(train_dir.glob("*.xml"))
    print(f"  PNG files: {len(pngs)}")
    print(f"  XML files: {len(xmls)}")

    if pngs:
        print(f"  Sample PNG names: {[p.name for p in pngs[:5]]}")
    if xmls:
        print(f"  Sample XML names: {[x.name for x in xmls[:5]]}")

    for xml_path in xmls[:max_files]:
        print(f"\n  --- RAW XML: {xml_path.name} (first 2000 chars) ---")
        raw = read_xml_text(xml_path)[:2000]
        print(raw)

        print(f"\n  --- PARSED STRUCTURE: {xml_path.name} ---")
        try:
            root = parse_xml_auto(xml_path)
            print(f"  Root tag: {root.tag}, attribs: {root.attrib}")
            for child in root:
                print(f"    Child tag: {child.tag}, attribs: {child.attrib}, text: {repr(child.text)}")
                for sub in child:
                    print(f"      Sub tag: {sub.tag}, attribs: {sub.attrib}, text: {repr(sub.text)}")
                    for subsub in sub:
                        print(f"        SubSub tag: {subsub.tag}, attribs: {subsub.attrib}, text: {repr(subsub.text)}")
                    if child.tag.lower() == "page" or root.tag.lower() == "page":
                        break
                break

            chars = root.findall(".//char")
            if not chars:
                chars = root.findall(".//Char")
            if not chars:
                for elem in root.iter():
                    if "char" in elem.tag.lower():
                        chars.append(elem)

            print(f"  Total <char> elements found: {len(chars)}")
            for c in chars[:5]:
                print(f"    char tag={c.tag}, attribs={c.attrib}")
                text_el = c.find("text") or c.find("Text")
                pos_el = c.find("position") or c.find("Position")
                print(f"      text element: tag={text_el.tag if text_el is not None else None}, "
                      f"text={repr(text_el.text) if text_el is not None else None}")
                print(f"      position element: tag={pos_el.tag if pos_el is not None else None}, "
                      f"text={repr(pos_el.text) if pos_el is not None else None}")

        except Exception as e:
            print(f"  PARSE ERROR: {e}")


def explore_hust_obc(hust_dir: Path):
    print("\n" + "=" * 70)
    print(f"[2] HUST-OBC: {hust_dir}")
    print("=" * 70)

    if not hust_dir.exists():
        print("  ERROR: directory does not exist!")
        return

    top_items = sorted(os.listdir(hust_dir))
    print(f"  Top-level items: {top_items[:20]}")

    deciphered = hust_dir / "deciphered"
    undeciphered = hust_dir / "undeciphered"

    for category_dir, label in [(deciphered, "deciphered"), (undeciphered, "undeciphered")]:
        print(f"\n  --- {label.upper()} ---")
        if not category_dir.exists():
            print(f"    Not found at {category_dir}")
            continue

        subdirs = sorted([d for d in category_dir.iterdir() if d.is_dir()])
        print(f"    Subdirectory count: {len(subdirs)}")
        if subdirs:
            print(f"    First 5: {[d.name for d in subdirs[:5]]}")

        sample_dirs = subdirs[:3] if subdirs else []
        for sd in sample_dirs:
            inner_items = sorted(os.listdir(sd))
            print(f"\n    [{sd.name}] contents ({len(inner_items)} items): {inner_items[:15]}")

            inner_subdirs = [d for d in sd.iterdir() if d.is_dir()]
            if inner_subdirs:
                for isd in inner_subdirs[:2]:
                    isd_items = sorted(os.listdir(isd))
                    print(f"      [{isd.name}] contents ({len(isd_items)} items): {isd_items[:10]}")

                    for item in isd_items[:3]:
                        fpath = isd / item
                        if fpath.suffix == ".json":
                            print(f"        JSON preview ({item}):")
                            with open(fpath, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                print(f"          List of {len(data)} items, first: {data[0] if data else 'empty'}")
                            elif isinstance(data, dict):
                                print(f"          Keys: {list(data.keys())[:10]}")
                                for k in list(data.keys())[:3]:
                                    print(f"          [{k}]: {repr(data[k])[:200]}")
            else:
                for item in inner_items[:5]:
                    fpath = sd / item
                    if fpath.suffix == ".json":
                        print(f"      JSON preview ({item}):")
                        with open(fpath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            print(f"        List of {len(data)} items, first: {data[0] if data else 'empty'}")
                        elif isinstance(data, dict):
                            print(f"        Keys: {list(data.keys())[:10]}")
                            for k in list(data.keys())[:3]:
                                print(f"        [{k}]: {repr(data[k])[:200]}")
                    elif fpath.is_file():
                        print(f"      File: {item} ({fpath.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Explore data formats on server")
    parser.add_argument("--train_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/train/out_of_domain")
    parser.add_argument("--hust_dir", type=str,
                        default="/home/apulis-dev/userdata/lbh/danc/HUST-OBC")
    args = parser.parse_args()

    explore_train_xml(Path(args.train_dir))
    explore_hust_obc(Path(args.hust_dir))

    print("\n" + "=" * 70)
    print("EXPLORATION COMPLETE. Check output above for format details.")
    print("=" * 70)


if __name__ == "__main__":
    main()
