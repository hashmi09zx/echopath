"""
EchoPath - COCO Mobility Subset Preparation Utility (Phase 1: Object Detection)

This script processes a standard MS COCO dataset (or custom COCO-formatted annotations),
filters annotations to mobility-relevant classes, converts bounding boxes to YOLO format,
and outputs a ready-to-train YOLO dataset structure along with a dataset configuration YAML file.

Default Mobility Classes:
  person, car, bicycle, motorcycle, bus, truck, dog, chair, bench, door, traffic_sign, pole, bottle, table, bed

Expected Source COCO Structure:
  <coco_dir>/
  ├── annotations/
  │   ├── instances_train2017.json
  │   └── instances_val2017.json
  ├── train2017/ (images)
  └── val2017/   (images)

Output YOLO Dataset Structure:
  <output_dir>/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── mobility_coco.yaml
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Default mobility classes requested for EchoPath
DEFAULT_MOBILITY_CLASSES = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "dog",
    "chair",
    "bench",
    "door",
    "traffic_sign",
    "pole",
    "bottle",
    "table",
    "bed",
]

# Standard COCO category name alias mapping (e.g., 'dining table' -> 'table')
COCO_NAME_ALIASES = {
    "dining table": "table",
    "stop sign": "traffic_sign",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter COCO dataset for EchoPath mobility-relevant classes and export to YOLO format."
    )
    parser.add_argument(
        "--coco-dir",
        type=str,
        default="data/coco",
        help="Path to raw source COCO directory (default: data/coco)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/mobility_coco",
        help="Path to save processed YOLO format dataset (default: data/mobility_coco)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_MOBILITY_CLASSES,
        help="List of class names to extract (default: mobility-relevant classes)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If > 0, limit processing to N sample images per split for local dev testing (default: 0 = full dataset)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect annotations and print summary without copying images or writing label files.",
    )
    return parser.parse_args()


def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """Converts COCO bbox [x_min, y_min, width, height] to YOLO normalized format [x_center, y_center, w, h]."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_width
    y_center = (y_min + h / 2.0) / img_height
    norm_w = w / img_width
    norm_h = h / img_height
    return (
        max(0.0, min(1.0, x_center)),
        max(0.0, min(1.0, y_center)),
        max(0.0, min(1.0, norm_w)),
        max(0.0, min(1.0, norm_h)),
    )


def process_split(split_name, json_path, img_dir, out_dir, target_classes, sample_limit=0, dry_run=False):
    """Processes a single dataset split (train or val)."""
    if not json_path.exists():
        print(f"Skipping split '{split_name}': annotation file '{json_path}' not found.")
        return 0

    print(f"\nProcessing split '{split_name}' from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Build category ID to target class index mapping
    target_class_set = set(target_classes)
    cat_id_to_class_idx = {}
    matched_categories = []

    for cat in coco_data.get("categories", []):
        raw_name = cat["name"]
        normalized_name = COCO_NAME_ALIASES.get(raw_name, raw_name)
        if normalized_name in target_class_set:
            class_idx = target_classes.index(normalized_name)
            cat_id_to_class_idx[cat["id"]] = class_idx
            matched_categories.append((cat["id"], raw_name, normalized_name, class_idx))

    print(f"Found {len(matched_categories)} matching COCO categories out of {len(target_classes)} target mobility classes.")

    if not cat_id_to_class_idx:
        print("WARNING: No matching categories found in annotations for target classes!")
        return 0

    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in coco_data.get("annotations", []):
        cat_id = ann.get("category_id")
        if cat_id in cat_id_to_class_idx and ann.get("iscrowd", 0) == 0:
            img_id = ann["image_id"]
            img_id_to_anns.setdefault(img_id, []).append(ann)

    images = [img for img in coco_data.get("images", []) if img["id"] in img_id_to_anns]
    print(f"Found {len(images)} images containing target mobility objects.")

    if sample_limit > 0:
        images = images[:sample_limit]
        print(f"Applying --sample filter: Processing first {len(images)} images for testing.")

    if dry_run:
        print(f"[Dry Run] Split '{split_name}' ready. Would export {len(images)} images and associated YOLO label files.")
        return len(images)

    # Create destination directories
    out_img_dir = out_dir / "images" / split_name
    out_lbl_dir = out_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    exported_count = 0
    for img_info in images:
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        src_img_path = img_dir / file_name
        if not src_img_path.exists():
            continue

        # Copy image file
        dst_img_path = out_img_dir / file_name
        if not dst_img_path.exists():
            shutil.copy2(src_img_path, dst_img_path)

        # Write YOLO label file
        label_file_name = Path(file_name).stem + ".txt"
        label_path = out_lbl_dir / label_file_name

        anns = img_id_to_anns.get(img_id, [])
        yolo_lines = []
        for ann in anns:
            cat_idx = cat_id_to_class_idx[ann["category_id"]]
            bbox = ann["bbox"]  # [x, y, w, h]
            cx, cy, nw, nh = convert_coco_bbox_to_yolo(bbox, img_w, img_h)
            yolo_lines.append(f"{cat_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(label_path, "w", encoding="utf-8") as f_lbl:
            f_lbl.write("\n".join(yolo_lines) + "\n")

        exported_count += 1

    print(f"Successfully exported {exported_count} images and label files to '{out_dir}'.")
    return exported_count


def generate_dataset_yaml(output_dir, target_classes):
    """Generates the dataset YAML file required by Ultralytics YOLO."""
    output_dir = Path(output_dir).resolve()
    yaml_path = output_dir / "mobility_coco.yaml"

    yaml_content = f"""# EchoPath Mobility-Relevant COCO Subset Dataset Config
path: {output_dir}
train: images/train
val: images/val

names:
"""
    for idx, cls_name in enumerate(target_classes):
        yaml_content += f"  {idx}: {cls_name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nGenerated YOLO dataset configuration YAML at: {yaml_path}")


def main():
    args = parse_args()
    coco_dir = Path(args.coco_dir)
    output_dir = Path(args.output_dir)

    print("=== EchoPath COCO Subset Preparation Utility ===")
    print(f"Source COCO Dir  : {coco_dir}")
    print(f"Output Directory : {output_dir}")
    print(f"Target Classes   : {args.classes}")
    print(f"Sample Limit     : {args.sample if args.sample > 0 else 'Full'}")
    print(f"Dry Run Mode     : {args.dry_run}")
    print("================================================")

    if not coco_dir.exists() and not args.dry_run:
        print(f"\nNOTE: Source COCO directory '{coco_dir}' does not exist.")
        print("To process raw COCO dataset later, download COCO train2017/val2017 and annotations into:")
        print(f"  {coco_dir}/annotations/instances_train2017.json")
        print(f"  {coco_dir}/train2017/")
        print(f"  {coco_dir}/val2017/\n")
        print("Creating placeholder YOLO dataset config file for inspection...")
        output_dir.mkdir(parents=True, exist_ok=True)
        generate_dataset_yaml(output_dir, args.classes)
        return

    # Process train split
    process_split(
        split_name="train",
        json_path=coco_dir / "annotations" / "instances_train2017.json",
        img_dir=coco_dir / "train2017",
        out_dir=output_dir,
        target_classes=args.classes,
        sample_limit=args.sample,
        dry_run=args.dry_run,
    )

    # Process val split
    process_split(
        split_name="val",
        json_path=coco_dir / "annotations" / "instances_val2017.json",
        img_dir=coco_dir / "val2017",
        out_dir=output_dir,
        target_classes=args.classes,
        sample_limit=args.sample,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        generate_dataset_yaml(output_dir, args.classes)


if __name__ == "__main__":
    main()
