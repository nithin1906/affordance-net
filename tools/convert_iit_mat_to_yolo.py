# tools/convert_iit_mat_to_yolo.py
"""
Convert IIT_Affordances_2017 dataset (.mat annotations) to YOLO-style labels.
Usage:
  python tools/convert_iit_mat_to_yolo.py --src "D:/Projects/Datasets/IIT_Affordances_2017" --out "./affordance_dataset" --split 0.8
"""

import argparse
import os
import shutil
from pathlib import Path
import json
import math
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="IIT dataset root (contains images + annotation .mat)")
    p.add_argument("--out", required=True, help="output base folder (will create images/, labels/)")
    p.add_argument("--split", type=float, default=0.8, help="train split fraction")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    return p.parse_args()

def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    images_out = out / "images"
    labels_out = out / "labels"
    ensure_dir(images_out)
    ensure_dir(labels_out)

    # --- Attempt to find annotation .mat files (common IIT layout uses 'dataset' .mat) ---
    mat_paths = list(src.rglob("*.mat"))
    if not mat_paths:
        print("No .mat files found. Make sure IIT_Affordances_2017 root is correct.")
        return
    # pick the first mat that contains annotation arrays
    mat = loadmat(str(mat_paths[0]), squeeze_me=True, struct_as_record=False)
    # Heuristic: IIT_Affordances often stores an array 'images' or 'dataset'
    # We try a few keys to find entries with filename + bounding boxes + class ids
    entries = None
    for key in ("images", "dataset", "imdb", "annolist", "imageList"):
        if key in mat:
            entries = mat[key]
            break
    if entries is None:
        # Try scanning mat for arrays with 'filename' attribute
        for k,v in mat.items():
            if hasattr(v, "__array_struct__") or isinstance(v, (list, tuple)):
                entries = v
                break

    # Fallback: list image files and create empty labels if no annotations found
    image_files = list(src.rglob("*.jpg")) + list(src.rglob("*.png")) + list(src.rglob("*.jpeg"))
    if not entries:
        print("No structured annotations found in .mat — creating YOLO placeholders for images")
        classes = set()
        # copy images into images/
        for im in tqdm(image_files, desc="copy images"):
            dst = images_out / im.name
            if not dst.exists():
                shutil.copy2(im, dst)
        # empty labels directory created; classes file empty
        (out / "classes.txt").write_text("")
        print("Wrote empty classes.txt; you can edit it manually.")
        return

    # For a real IIT .mat layout we need to inspect structure - common fields:
    # entries[i].filename (string)
    # entries[i].objects[j].bbox = [x1,y1,w,h] or [x1,y1,x2,y2]
    # entries[i].objects[j].class / label
    # We'll attempt to be flexible by introspecting attributes.
    data = []
    classes_set = set()

    def get_attr(obj, name):
        return getattr(obj, name, None)

    # If entries is array-like
    if not isinstance(entries, (list, tuple)):
        try:
            entries = list(entries)
        except Exception:
            entries = [entries]

    for item in tqdm(entries, desc="scanning annotations"):
        # locate filename
        filename = None
        for attr in ("filename", "name", "imgname", "image"):
            val = get_attr(item, attr)
            if val:
                filename = str(val)
                break
        if not filename:
            # some IIT structures keep path in item.file.name etc.
            if hasattr(item, "file"):
                filename = getattr(item.file, "name", None)
        if not filename:
            continue
        # find bounding objects
        objs = []
        for attr in ("objects", "bbox", "boxes", "gtboxes", "annots"):
            v = get_attr(item, attr)
            if v:
                objs = v
                break
        # ensure objs is iterable
        if objs is None:
            objs = []
        if not isinstance(objs, (list, tuple)):
            objs = [objs]

        # copy image src -> out/images/
        # try to find path by matching file name under src
        possible = list(src.rglob(os.path.basename(filename)))
        if possible:
            chosen_im = possible[0]
            dst_image = images_out / chosen_im.name
            if not dst_image.exists():
                shutil.copy2(chosen_im, dst_image)
            w,h = Image.open(chosen_im).size
        else:
            # image might have absolute path stored in filename
            if os.path.isabs(filename) and os.path.exists(filename):
                chosen_im = Path(filename)
                dst_image = images_out / chosen_im.name
                if not dst_image.exists():
                    shutil.copy2(chosen_im, dst_image)
                w,h = Image.open(chosen_im).size
            else:
                # skip if no image found
                continue

        # create label lines for this image
        lines = []
        for o in objs:
            # try multiple shapes for bbox
            bbox = None
            clsname = None
            # look for bounding box fields
            for battr in ("bbox","box","rect","bndbox","bb"):
                b = get_attr(o, battr)
                if b is not None:
                    bbox = b
                    break
            if bbox is None:
                # if o is a simple struct with x1,y1,x2,y2
                xs = [get_attr(o, x) for x in ("x1","y1","x2","y2")]
                if all(v is not None for v in xs):
                    bbox = xs
            # class
            for cat in ("class","label","classname","obj_class","category"):
                c = get_attr(o, cat)
                if c:
                    clsname = str(c)
                    break
            if clsname is None:
                # sometimes class is numeric id
                cid = get_attr(o, "cid") or get_attr(o, "class_id")
                if cid is not None:
                    clsname = str(int(cid))
            if bbox is None or clsname is None:
                continue

            # Normalize bbox to YOLO format (x_center y_center w h) relative
            # Accept either [x,y,w,h] or [x1,y1,x2,y2]
            arr = list(bbox) if not isinstance(bbox, (str,bytes)) else []
            if len(arr) >= 4:
                x0,y0,x1,y1 = None,None,None,None
                # detect format
                if arr[2] > arr[0] and arr[3] > arr[1] and arr[2] - arr[0] < max(arr[2],arr[3]):
                    # either x,y,w,h or x1,y1,x2,y2; decide by magnitude:
                    # If arr[2] is width (smaller than image width) it's ambiguous; we assume:
                    # if arr[2] - arr[0] > 0 and arr[2] > arr[0] and arr[3] > arr[1] and arr[2] - arr[0] > 1:
                    # fallback: treat as [x1,y1,x2,y2] if arr[2] > arr[0] and arr[3] > arr[1]
                    x1 = float(arr[0])
                    y1 = float(arr[1])
                    x2 = float(arr[2])
                    y2 = float(arr[3])
                    # If values look like x,y,w,h rather than x1,x2
                    if x2 - x1 > w or y2 - y1 > h:
                        # likely x,y,w,h
                        x1 = float(arr[0])
                        y1 = float(arr[1])
                        x2 = x1 + float(arr[2])
                        y2 = y1 + float(arr[3])
                else:
                    # fallback
                    continue

                xc = (x1 + x2) / 2.0 / w
                yc = (y1 + y2) / 2.0 / h
                ww = (x2 - x1) / w
                hh = (y2 - y1) / h
                # clamp
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                ww = min(max(ww, 0.0), 1.0)
                hh = min(max(hh, 0.0), 1.0)

                lines.append((clsname, xc, yc, ww, hh))

            # collect class set
            classes_set.add(str(clsname))

        # write label file
        if lines:
            label_path = labels_out / (dst_image.name.rsplit(".",1)[0] + ".txt")
            with open(label_path, "w", encoding="utf-8") as f:
                for clsname, xc, yc, ww, hh in lines:
                    f.write(f"{clsname} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

    # write classes.txt (sorted)
    classes_list = sorted(list(classes_set), key=lambda x: x)
    # if numeric labels (like '0','1') it's better to map to strings - we keep as-is
    classes_file = out / "classes.txt"
    classes_file.write_text("\n".join(classes_list), encoding="utf-8")
    print(f"Written classes.txt with {len(classes_list)} classes -> {classes_file}")

    # make small train/val split by moving label/image file pairs (simple)
    # collect images in images_out
    imgs = list(images_out.iterdir())
    import random
    random.seed(args.seed)
    random.shuffle(imgs)
    n = len(imgs)
    ntrain = int(n * args.split)
    train = imgs[:ntrain]
    val = imgs[ntrain:]
    for subset, files in [("train", train), ("val", val)]:
        img_dst = out / "images" / subset
        lbl_dst = out / "labels" / subset
        ensure_dir(img_dst)
        ensure_dir(lbl_dst)
        for im in files:
            src_img = images_out / im.name
            dst_img = img_dst / im.name
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            lbl_src = labels_out / (im.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst / lbl_src.name)
    print("Split into train/val under images/ and labels/")

    # write default class->affordance mapping placeholder
    default_map = {c: ["unknown"] for c in classes_list}
    map_file = out / "class_to_affordance.json"
    map_file.write_text(json.dumps(default_map, indent=2), encoding="utf-8")
    # write affordance_data.yaml (for yolov8/other pipelines)
    yaml_content = {
        "train": str((out / "images" / "train").as_posix()),
        "val": str((out / "images" / "val").as_posix()),
        "nc": len(classes_list),
        "names": classes_list
    }
    import yaml
    (out / "affordance_data.yaml").write_text(yaml.safe_dump(yaml_content), encoding="utf-8")
    print("Wrote affordance_data.yaml and class_to_affordance.json")

if __name__ == "__main__":
    main()
