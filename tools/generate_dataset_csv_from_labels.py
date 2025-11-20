#!/usr/bin/env python3
"""
tools/generate_dataset_csv_from_labels.py

Generates affordance_crops/dataset.csv from:
 - YOLO labels in affordance_dataset/labels/*.txt
 - Images in affordance_dataset/images/*.jpg|png
 - class_to_affordance.json or classes.txt mapping

CSV format produced:
image,affordances
C:\abs\path\to\crop_0001.jpg,grasp|pour

This script creates a crops folder (affordance_crops/images) by cropping each
bounding box in the labels. It avoids re-cropping if the crop already exists.
"""

import argparse, os, json, csv
from pathlib import Path
from PIL import Image

def load_classes(classes_txt):
    if classes_txt and Path(classes_txt).exists():
        with open(classes_txt, 'r', encoding='utf-8') as f:
            return [l.strip() for l in f if l.strip()]
    return []

def load_affordance_map(map_path):
    if map_path and Path(map_path).exists():
        return json.loads(Path(map_path).read_text(encoding='utf-8'))
    return {}

def yolo_to_bbox(yolo, img_w, img_h):
    # yolo: class_idx x_center y_center width height (normalized)
    cls_id = int(float(yolo[0]))
    xc = float(yolo[1]) * img_w
    yc = float(yolo[2]) * img_h
    w = float(yolo[3]) * img_w
    h = float(yolo[4]) * img_h
    x1 = max(0, int(xc - w/2))
    y1 = max(0, int(yc - h/2))
    x2 = min(img_w, int(xc + w/2))
    y2 = min(img_h, int(yc + h/2))
    return cls_id, x1, y1, x2, y2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels_root", required=True)
    p.add_argument("--images_root", required=True)
    p.add_argument("--out_dir", default="./affordance_crops")
    p.add_argument("--map_file", default="./affordance_dataset/class_to_affordance.json")
    p.add_argument("--classes_txt", default="./affordance_dataset/classes.txt")
    p.add_argument("--min_area", type=int, default=100)
    args = p.parse_args()

    labels_root = Path(args.labels_root)
    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "images"
    crops_dir.mkdir(parents=True, exist_ok=True)

    afford_map = load_affordance_map(args.map_file)
    classes = load_classes(args.classes_txt)

    csv_path = out_dir / "dataset.csv"
    rows = []

    label_files = list(labels_root.rglob("*.txt"))
    print("Found label files:", len(label_files))

    crop_idx = 0
    for lf in label_files:
        # match image name
        img_rel = lf.with_suffix(".jpg").name
        # try jpg and png
        img_path = images_root / lf.with_suffix(".jpg").name
        if not img_path.exists():
            img_path = images_root / lf.with_suffix(".png").name
        if not img_path.exists():
            # maybe nested structure: try same relative path
            candidate = images_root / lf.relative_to(labels_root)
            candidate = candidate.with_suffix(".jpg")
            if candidate.exists():
                img_path = candidate
            else:
                print("Image not found for label:", lf)
                continue

        img = Image.open(img_path).convert("RGB")
        w,h = img.size

        with open(lf, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) == 0:
            continue

        # gather affordance strings for this image (we will create one crop file per bbox)
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, x1, y1, x2, y2 = yolo_to_bbox(parts, w, h)
            area = (x2-x1)*(y2-y1)
            if area < args.min_area:
                continue
            # crop
            crop_idx += 1
            crop_name = f"crop_{crop_idx:06d}.jpg"
            crop_path = crops_dir / crop_name
            if not crop_path.exists():
                crop = img.crop((x1, y1, x2, y2))
                crop.save(crop_path, quality=95)
            # map class->affordance (class_to_affordance.json could map one class to list)
            aff = afford_map.get(str(cls_id)) or afford_map.get(cls_id)
            if aff is None:
                # try classes.txt lookup
                if classes and cls_id < len(classes):
                    cls_name = classes[cls_id]
                    # default: affordance same as class name
                    aff = [cls_name]
                else:
                    aff = []
            # ensure it's a list of strings -> join with |
            if isinstance(aff, str):
                aff_list = [aff]
            elif isinstance(aff, (list,tuple)):
                aff_list = [str(x) for x in aff]
            else:
                aff_list = []
            rows.append([str(crop_path.resolve()), "|".join(aff_list)])

    # write CSV
    print(f"Writing {len(rows)} rows to {csv_path}")
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(["image","affordances"])
        for r in rows:
            w.writerow(r)

    print("Done. crops saved to:", str(crops_dir))
    print("CSV saved to:", str(csv_path))

if __name__ == "__main__":
    main()
