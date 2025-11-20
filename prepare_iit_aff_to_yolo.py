# prepare_iit_aff_to_yolo.py
"""
Usage examples:

# COCO:
python prepare_iit_aff_to_yolo.py --src /path/to/iit_aff --format coco --coco_json /path/to/annotations.json --out ./affordance_dataset --train 0.8 --val 0.1

# VOC (XMLs next to images):
python prepare_iit_aff_to_yolo.py --src /path/to/iit_aff/images --format voc --out ./affordance_dataset

# simple CSV:
python prepare_iit_aff_to_yolo.py --src /path/to/images --format simple --simple_csv annotations.csv --out ./affordance_dataset

"""

import os
import argparse
import json
import shutil
import math
import random
from pathlib import Path
from collections import defaultdict
from xml.etree import ElementTree as ET

try:
    from PIL import Image
except Exception:
    raise SystemExit("Pillow required: pip install pillow")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def bbox_from_polygon(poly):
    # poly: [x1,y1,x2,y2,...]
    xs = poly[0::2]
    ys = poly[1::2]
    xmin = min(xs); xmax = max(xs)
    ymin = min(ys); ymax = max(ys)
    return xmin, ymin, xmax, ymax

def write_yolo_label(label_path, items):
    # items: list of (class_id, xcenter_n, ycenter_n, w_n, h_n)
    with open(label_path, "w") as f:
        for it in items:
            f.write(f"{it[0]} {it[1]:.6f} {it[2]:.6f} {it[3]:.6f} {it[4]:.6f}\n")

def normalize_bbox(xmin,ymin,xmax,ymax, img_w, img_h):
    # convert to x_center,y_center,w,h (normalized)
    w = xmax - xmin
    h = ymax - ymin
    cx = xmin + w/2.0
    cy = ymin + h/2.0
    return cx/img_w, cy/img_h, w/img_w, h/img_h

def parse_coco(coco_json):
    # returns dict image_id -> filename, and list of annotations
    with open(coco_json, "r") as f:
        coco = json.load(f)
    images = {img["id"]: img for img in coco.get("images", [])}
    anns = coco.get("annotations", [])
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    return images, anns, categories

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        cname = obj.find('name').text
        bnd = obj.find('bndbox')
        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)
        # optional affordances as <affordances><aff>grasp</aff>...</affordances>
        affs = []
        aff_node = obj.find('affordances')
        if aff_node is not None:
            affs = [a.text for a in aff_node.findall('aff')]
        objects.append({"class": cname, "bbox": [xmin,ymin,xmax,ymax], "affordances": affs})
    return w,h,objects

def parse_simple_csv(csv_path):
    # expects columns: image,xmin,ymin,xmax,ymax,class,aff1|aff2|...
    rows = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            image = parts[0].strip()
            xmin = float(parts[1]); ymin = float(parts[2]); xmax = float(parts[3]); ymax = float(parts[4])
            cname = parts[5].strip()
            affs = []
            if len(parts) > 6:
                affs = parts[6].strip().split("|")
            rows[image].append({"class": cname, "bbox":[xmin,ymin,xmax,ymax], "affordances": affs})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source images root (or dataset root)")
    ap.add_argument("--format", required=True, choices=["coco","voc","simple"], help="annotation format")
    ap.add_argument("--coco_json", help="path to COCO annotations JSON")
    ap.add_argument("--simple_csv", help="path to simple CSV annotations")
    ap.add_argument("--out", required=True, help="output folder for YOLO dataset")
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    src = Path(args.src)
    out = Path(args.out)
    ensure_dir(out)

    # collect annotations into image -> objects mapping
    image_objects = defaultdict(list)
    classes_set = set()
    aff_set = set()

    if args.format == "coco":
        if not args.coco_json:
            raise SystemExit("COCO JSON path required with --coco_json")
        images, anns, categories = parse_coco(args.coco_json)
        # map category id -> name
        for ann in anns:
            img = images.get(ann["image_id"])
            if img is None: continue
            fname = img["file_name"]
            # bbox could be [x,y,w,h] in COCO
            if "bbox" in ann and len(ann["bbox"])==4:
                x,y,w,h = ann["bbox"]
                xmin,ymin,xmax,ymax = x, y, x+w, y+h
            elif "segmentation" in ann and ann["segmentation"]:
                seg = ann["segmentation"][0]
                xmin,ymin,xmax,ymax = bbox_from_polygon(seg)
            else:
                continue
            cname = categories.get(ann["category_id"], str(ann.get("category_id")))
            affs = ann.get("affordances", []) if isinstance(ann.get("affordances", []), list) else []
            image_objects[fname].append({"class": cname, "bbox":[xmin,ymin,xmax,ymax], "affordances": affs})
            classes_set.add(cname)
            for a in affs: aff_set.add(a)

    elif args.format == "voc":
        # expects images in src and XML annotations in same folder or in src/Annotations
        # scan for xml files
        xml_paths = list(src.rglob("*.xml"))
        if not xml_paths:
            possible = list((src/"Annotations").rglob("*.xml"))
            xml_paths = possible
        if not xml_paths:
            raise SystemExit("No XML files found for VOC parsing.")
        for xp in xml_paths:
            w,h,objects = parse_voc_xml(xp)
            imgname = xp.with_suffix('.jpg').name
            # allow png too:
            if not (src/imgname).exists():
                imgname = xp.with_suffix('.png').name
            for o in objects:
                image_objects[imgname].append(o)
                classes_set.add(o["class"])
                for a in o.get("affordances",[]): aff_set.add(a)

    else: # simple
        if not args.simple_csv:
            raise SystemExit("simple format requires --simple_csv")
        rows = parse_simple_csv(args.simple_csv)
        for image, objs in rows.items():
            for o in objs:
                image_objects[image].append(o)
                classes_set.add(o["class"])
                for a in o.get("affordances",[]): aff_set.add(a)

    # sort classes and affordances to stable indices
    classes = sorted(list(classes_set))
    affordances = sorted(list(aff_set))

    # write classes and affordances
    with open(out/"classes.txt", "w", encoding="utf-8") as f:
        for c in classes:
            f.write(c + "\n")
    with open(out/"affordances.json", "w", encoding="utf-8") as f:
        json.dump({"affordances": affordances}, f, indent=2)

    # gather image list that exist on disk
    image_files = []
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg",".jpeg",".png"]:
            image_files.append(p)
    name_to_path = {p.name: p for p in image_files}

    # final images that have annotations and exist on disk
    annotated_images = [name for name in image_objects.keys() if name in name_to_path]
    if not annotated_images:
        print("Warning: no annotated images matched image files in", src)
    random.shuffle(annotated_images)

    # splits
    n = len(annotated_images)
    n_train = int(n * args.train)
    n_val = int(n * args.val)
    train_list = annotated_images[:n_train]
    val_list = annotated_images[n_train:n_train+n_val]
    test_list = annotated_images[n_train+n_val:]

    # make directories
    for d in ["images/train","images/val","images/test","labels/train","labels/val","labels/test"]:
        ensure_dir(out/d)

    # copy images & write labels
    cls_to_id = {c:i for i,c in enumerate(classes)}
    def copy_and_label(name, split):
        srcp = name_to_path.get(name)
        if srcp is None:
            return
        dst_img = out/f"images/{split}"/name
        shutil.copy2(srcp, dst_img)
        # open to get width/height
        im = Image.open(srcp)
        iw, ih = im.size
        items = []
        for obj in image_objects[name]:
            xmin,ymin,xmax,ymax = obj["bbox"]
            # clamp to image bounds:
            xmin = max(0, min(xmin, iw-1))
            ymin = max(0, min(ymin, ih-1))
            xmax = max(0, min(xmax, iw-1))
            ymax = max(0, min(ymax, ih-1))
            cxn, cyn, wn, hn = normalize_bbox(xmin,ymin,xmax,ymax, iw, ih)
            cid = cls_to_id[obj["class"]]
            items.append((cid, cxn, cyn, wn, hn))
        label_path = out/f"labels/{split}"/(Path(name).stem + ".txt")
        write_yolo_label(label_path, items)

    for nm in train_list:
        copy_and_label(nm, "train")
    for nm in val_list:
        copy_and_label(nm, "val")
    for nm in test_list:
        copy_and_label(nm, "test")

    # write data.yaml
    data_yaml = {
        "train": str((out/"images/train").resolve()),
        "val": str((out/"images/val").resolve()),
        "test": str((out/"images/test").resolve()),
        "nc": len(classes),
        "names": classes
    }
    with open(out/"data.yaml","w", encoding="utf-8") as f:
        yaml = json.dumps(data_yaml, indent=2)
        # data.yaml in YOLO expects YAML; write simple YAML style:
        f.write(f"train: {data_yaml['train']}\n")
        f.write(f"val: {data_yaml['val']}\n")
        f.write(f"test: {data_yaml['test']}\n")
        f.write(f"nc: {data_yaml['nc']}\n")
        f.write("names:\n")
        for i,nm in enumerate(classes):
            f.write(f"  {i}: '{nm}'\n")

    print("Done. Output in:", out)
    print("Classes:", classes)
    print("Affordances:", affordances)
    print("Train/Val/Test:", len(train_list), len(val_list), len(test_list))

if __name__ == "__main__":
    main()
