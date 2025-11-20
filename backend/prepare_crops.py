# backend/prepare_crops.py
import argparse
from pathlib import Path
from PIL import Image
import os
import csv
import json
from tqdm import tqdm

#
# ----- THIS IS THE FIX -----
#
# I have replaced the old, small dictionary with our
# comprehensive 26-class mapping. This will create
# a rich dataset for your classifier.
CLASS_TO_AFF = {
    # Graspable / Holdable
    "bottle": ["Graspable", "Containment", "Pourable"],
    "cup": ["Graspable", "Containment", "Pourable"],
    "fork": ["Graspable", "Penetrable"],
    "knife": ["Graspable", "Cutting"],
    "spoon": ["Graspable", "Containment"],
    "cell phone": ["Graspable", "Pressable", "Visibility"],
    "laptop": ["Graspable", "Openable", "Pressable", "Visibility"],
    "mouse": ["Graspable", "Pressable", "Rotatable"],
    "remote": ["Graspable", "Pressable"],
    "keyboard": ["Pressable"],
    "book": ["Graspable", "Openable", "Visibility"],
    "backpack": ["Graspable", "Containment", "Wearable", "Openable"],
    "handbag": ["Graspable", "Containment", "Wearable", "Openable"],
    "umbrella": ["Graspable", "Protective"],
    "scissors": ["Graspable", "Cutting"],
    "toothbrush": ["Graspable", "Cleaning"],

    # Sittable / Support
    "chair": ["Sittable", "Support", "Graspable"],
    "couch": ["Sittable", "Support"],
    "bench": ["Sittable", "Support"],
    "dining table": ["Support"],
    "bed": ["Sittable", "Support"],

    # Containment / Support
    "bowl": ["Containment", "Support", "Graspable"],
    "potted plant": ["Containment", "Support"],
    "vase": ["Containment", "Support", "Graspable"],

    # Visibility / Other
    "tv": ["Visibility"],
    "person": ["Wearable"], # (e.g., for detecting clothes, backpack)
}
#
# ----- END OF FIX -----
#

def yolo_to_bbox(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w/2) * img_w)
    y1 = int((yc - h/2) * img_h)
    x2 = int((xc + w/2) * img_w)
    y2 = int((yc + h/2) * img_h)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w, x2); y2 = min(img_h, y2)
    return x1, y1, x2, y2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True, help="images root")
    p.add_argument("--labels", required=True, help="yolo labels root")
    p.add_argument("--out", default="./affordance_crops", help="output dir for crops and csv")
    p.add_argument("--classnames", help="path to data.yaml or classes.txt")
    p.add_argument("--padding", type=float, default=0.1, help="padding around bbox (e.g. 0.1 = 10%)")
    args = p.parse_args()

    img_root = Path(args.images)
    lbl_root = Path(args.labels)
    out_root = Path(args.out)
    
    crops_dir = out_root / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "dataset.csv"

    classnames = []
    if args.classnames:
        cn_path = Path(args.classnames)
        if cn_path.suffix == ".yaml":
            try:
                import yaml
                data = yaml.safe_load(cn_path.read_text())
                classnames = data.get("names", [])
            except Exception as e:
                print(f"Error reading YAML {cn_path}: {e}")
        elif cn_path.exists():
            classnames = [n.strip() for n in cn_path.read_text().splitlines() if n.strip()]
    
    if not classnames:
        print("Warning: No classnames loaded. Will use numeric class IDs and default affordances.")

    rows = []
    print(f"Scanning for labels in {lbl_root}...")
    
    label_files = list(lbl_root.rglob("*.txt"))
    if not label_files:
        print(f"Error: No label files (*.txt) found in {lbl_root}")
        return

    for lbl_path in tqdm(label_files, desc="Processing labels"):
        img_path = img_root / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = img_root / (lbl_path.stem + ".png") # try png
            if not img_path.exists():
                continue
        
        try:
            with Image.open(img_path) as img:
                iw, ih = img.width, img.height
        except Exception as e:
            print(f"Warning: Skipping corrupt image {img_path}: {e}")
            continue

        lines = lbl_path.read_text().splitlines()
        
        for i, line in enumerate(lines):
            try:
                toks = line.split()
                cls = int(toks[0])
                xc, yc, w, h = map(float, toks[1:5])
            except Exception:
                continue # skip bad line
            
            cname = classnames[cls].lower() if cls < len(classnames) else str(cls)
            
            # This is the FILTER. If the class is not in our
            # dictionary, we ignore it completely.
            if cname not in CLASS_TO_AFF:
                continue
                
            affs = CLASS_TO_AFF[cname]
            if not affs:
                continue # Skip if class has no affordances defined

            x1, y1, x2, y2 = yolo_to_bbox(xc, yc, w, h, iw, ih)
            
            # add padding
            pad_w = int((x2-x1) * args.padding)
            pad_h = int((y2-y1) * args.padding)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(iw, x2 + pad_w)
            y2 = min(ih, y2 + pad_h)

            crop_name = f"{img_path.stem}_{cls}_{i}.jpg"
            
            try:
                with Image.open(img_path) as img:
                    crop = img.crop((x1, y1, x2, y2))
                    crop.save(crops_dir / crop_name, quality=90)
            except Exception as e:
                print(f"Warning: Failed to save crop {crop_name}: {e}")
                continue

            # Add to our CSV row list
            rows.append([str(crops_dir / crop_name), ",".join(affs)])

    # write csv: image_path,aff1,aff2,...
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["image","affordances"])
        for r in rows:
            writer.writerow(r)

    print(f"Saved {len(rows)} crops to {crops_dir}")
    print(f"Saved CSV manifest to {csv_path}")

if __name__ == "__main__":
    main()