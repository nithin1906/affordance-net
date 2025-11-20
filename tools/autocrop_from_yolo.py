#!/usr/bin/env python3
"""
Autocrop images from YOLO-style labels and attach affordances based on class_to_affordance.json.
Outputs:
  - ./out_dir/crops/<image_basename>_idx.jpg
  - ./out_dir/dataset.csv  (image_rel_path, affordance1|affordance2|...)
"""

import argparse
from pathlib import Path
from PIL import Image
import json
import csv

def yolo_box_to_xyxy(box, w, h):
    # box: [x_center y_center width height] normalized
    xc, yc, bw, bh = map(float, box)
    x1 = (xc - bw/2) * w
    y1 = (yc - bh/2) * h
    x2 = (xc + bw/2) * w
    y2 = (yc + bh/2) * h
    return int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_root", "--images", required=True, help="root folder with images")
    p.add_argument("--labels_root", "--labels", required=True, help="root folder with YOLO .txt labels (same relative paths)")
    p.add_argument("--out_dir", "--out", default="./affordance_crops", help="output dir for crops + dataset.csv")
    p.add_argument("--map_file", required=True, help="class_to_affordance.json path")
    p.add_argument("--classes_txt", default=None, help="optional classes.txt to map class ids -> names")
    p.add_argument("--min_box_area", type=int, default=100, help="min crop area in px to keep")
    p.add_argument("--resize", type=int, default=0, help="if >0, resize crops to this size (square)")
    args = p.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # load mapping
    mapping = json.loads(Path(args.map_file).read_text(encoding="utf-8"))
    classes_list = None
    if args.classes_txt:
        classes_list = [ln.strip() for ln in Path(args.classes_txt).read_text(encoding="utf-8").splitlines() if ln.strip()]

    csv_rows = []
    count = 0
    images = list(images_root.glob("**/*.*"))
    images = [i for i in images if i.suffix.lower() in [".jpg",".jpeg",".png"]]

    for img_path in images:
        # find corresponding label file by relative path
        rel = img_path.relative_to(images_root)
        label_path = labels_root / rel.with_suffix(".txt")
        if not label_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            w,h = img.size
        except Exception as e:
            print("skipping image open error:", img_path, e)
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()
        for i,ln in enumerate(lines):
            parts = ln.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            box = parts[1:5]
            x1,y1,x2,y2 = yolo_box_to_xyxy(box, w, h)
            area = (x2-x1) * (y2-y1)
            if area < args.min_box_area:
                continue
            crop = img.crop((x1,y1,x2,y2))
            if args.resize and args.resize>0:
                crop = crop.resize((args.resize,args.resize))
            out_name = f"{rel.stem}_{i}.jpg"
            out_path = crops_dir / out_name
            crop.save(out_path, quality=90)
            # affordances by class
            cls_name = None
            if classes_list and cls < len(classes_list):
                cls_name = classes_list[cls]
            else:
                cls_name = f"class_{cls}"
            affs = mapping.get(cls_name, mapping.get(cls_name.lower(), ["grasp"]))
            csv_rows.append([str(out_path.relative_to(out)), "|".join(affs)])
            count += 1

    # write dataset.csv
    csv_path = out / "dataset.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image","affordances"])
        writer.writerows(csv_rows)

    print(f"Done. crops: {count}, csv: {csv_path}")

if __name__ == "__main__":
    main()
