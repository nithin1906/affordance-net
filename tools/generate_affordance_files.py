#!/usr/bin/env python3
"""
tools/generate_affordance_files.py

Usage:
  python tools/generate_affordance_files.py --src "D:/Projects/Datasets/IIT_Affordances_2017" --out "./affordance_dataset"

What it does:
1) Scans the dataset folder for images and YOLO-style label files (labels/{train,val,test} or same-folder .txt).
2) Collects unique class indices and/or names (if classes.txt present).
3) Writes:
   - ./affordance_dataset/class_to_affordance.json (merging defaults with detected classes)
   - ./affordance_dataset/data.yaml (affordance_data.yaml)
   - ./affordance_dataset/classes.txt (if missing)
4) Prints detected classes that were auto-mapped (so you can refine).
"""
import argparse
import json
from pathlib import Path
import re
from collections import defaultdict

DEFAULT_CLASS_TO_AFF = {
  "bottle": ["grasp", "pour"],
  "cup": ["grasp", "drink", "contain"],
  "mug": ["grasp", "drink", "contain"],
  "glass": ["grasp", "drink", "contain"],
  "bowl": ["grasp", "contain", "support"],
  "plate": ["grasp", "contain", "support"],
  "knife": ["grasp", "cut"],
  "fork": ["grasp", "eat"],
  "spoon": ["grasp", "eat"],
  "pen": ["grasp", "write", "point"],
  "pencil": ["grasp", "write", "point"],
  "phone": ["grasp", "use", "display"],
  "laptop": ["display", "use", "support"],
  "chair": ["support", "sit"],
  "table": ["support"],
  "sofa": ["support", "sit"],
  "box": ["grasp", "contain"],
  "book": ["grasp", "display", "read"],
  "bottle_cap": ["grasp", "open"],
  "container": ["grasp", "contain", "pour"],
  "unknown": ["grasp"]
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def find_images(root: Path):
    imgs = []
    for p in root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            imgs.append(p)
    return imgs

def find_yolo_labels_for_image(img_path: Path):
    # typical YOLO: same-name .txt next to image
    txt = img_path.with_suffix('.txt')
    if txt.exists():
        return txt
    # sometimes labels are in labels/ with same basename
    labels_dir = img_path.parent.parent / "labels"
    if labels_dir.exists():
        cand = labels_dir / img_path.name.replace(img_path.suffix, ".txt")
        if cand.exists():
            return cand
    return None

def detect_classes_from_label_file(label_file: Path):
    classes = set()
    try:
        for line in label_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line=line.strip()
            if not line:
                continue
            parts = line.split()
            # first token is class index in YOLO
            if re.match(r"^\d+$", parts[0]):
                classes.add(int(parts[0]))
            else:
                # some simple formats have class names directly
                classes.add(parts[0])
    except Exception as e:
        print(f"Warning: couldn't read {label_file}: {e}")
    return classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Root folder of IIT_AAFF dataset (images/labels/ ...)")
    parser.add_argument("--out", default="./affordance_dataset", help="Where to write generated files")
    parser.add_argument("--classes_txt", default=None, help="Optional existing classes.txt to use (path)")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("Scanning dataset root:", src)
    images = find_images(src)
    print("Found images:", len(images))

    # Collect label files and class tokens
    detected_class_tokens = set()
    # mapping from numeric index -> maybe names list (if classes.txt available)
    numeric_indices_found = set()
    labels_seen = 0
    for img in images:
        lbl = find_yolo_labels_for_image(img)
        if lbl and lbl.exists():
            labels_seen += 1
            tokens = detect_classes_from_label_file(lbl)
            for t in tokens:
                detected_class_tokens.add(t)
                if isinstance(t, int):
                    numeric_indices_found.add(t)

    print("Label files found for images:", labels_seen)
    # If classes.txt present in dataset or provided, read it
    classes_txt_path = None
    classes_list = []
    if args.classes_txt:
        classes_txt_path = Path(args.classes_txt)
    else:
        # check common places
        possible = [src / "classes.txt", src / "labels" / "classes.txt", src / "data" / "classes.txt"]
        for p in possible:
            if p.exists():
                classes_txt_path = p
                break

    if classes_txt_path and classes_txt_path.exists():
        print("Using classes file:", classes_txt_path)
        classes_list = [l.strip() for l in classes_txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    elif numeric_indices_found:
        max_idx = max(numeric_indices_found)
        print("Detected numeric label indices. No classes.txt found. Will create placeholder classes.txt for indices 0..", max_idx)
        classes_list = [f"class_{i}" for i in range(max_idx+1)]
    else:
        # maybe labels used names directly
        name_tokens = sorted([t for t in detected_class_tokens if not isinstance(t, int)])
        if name_tokens:
            print("Detected class names directly in labels. Using them.")
            classes_list = [str(t) for t in name_tokens]
        else:
            print("No label files found or no classes detected. Creating empty classes.txt placeholder.")
            classes_list = []

    # Build mapping class_name -> affordances using DEFAULT_CLASS_TO_AFF, add defaults for unknowns
    class_to_aff = {}
    unknowns = []
    for cname in classes_list:
        key = cname
        # normalize common variants
        key_norm = key.lower().replace(" ", "_")
        if key_norm in DEFAULT_CLASS_TO_AFF:
            class_to_aff[cname] = DEFAULT_CLASS_TO_AFF[key_norm]
        else:
            # try endswith based heuristics
            matched = False
            for dkey in DEFAULT_CLASS_TO_AFF:
                if dkey in key_norm:
                    class_to_aff[cname] = DEFAULT_CLASS_TO_AFF[dkey]
                    matched = True
                    break
            if not matched:
                class_to_aff[cname] = ["grasp"]  # conservative default
                unknowns.append(cname)

    # If classes_list empty, but we detected names in labels, use them
    if not classes_list and detected_class_tokens:
        # if tokens are ints -> we already created placeholders
        # else tokens are names -> convert to list
        inferred = [t for t in detected_class_tokens if not isinstance(t,int)]
        if inferred:
            for cname in inferred:
                key_norm = cname.lower().replace(" ", "_")
                if key_norm in DEFAULT_CLASS_TO_AFF:
                    class_to_aff[cname] = DEFAULT_CLASS_TO_AFF[key_norm]
                else:
                    class_to_aff[cname] = ["grasp"]
                    unknowns.append(cname)
            classes_list = list(class_to_aff.keys())

    # write classes.txt if missing
    classes_txt_out = out / "classes.txt"
    if not classes_txt_out.exists():
        print("Writing classes.txt to:", classes_txt_out)
        classes_txt_out.write_text("\n".join(classes_list), encoding="utf-8")
    else:
        print("classes.txt already exists at:", classes_txt_out)

    # write class_to_affordance.json
    cta_path = out / "class_to_affordance.json"
    print("Writing class_to_affordance.json to:", cta_path)
    cta_path.write_text(json.dumps(class_to_aff, indent=2), encoding="utf-8")

    # write affordance_data.yaml (basic)
    aff_yaml = out / "affordance_data.yaml"
    affordances_all = sorted({a for affs in class_to_aff.values() for a in affs})
    import yaml
    data_yaml = {
        "dataset_root": str(out).replace("\\","/"),
        "images": {
            "train": str((out / 'images' / 'train').resolve()).replace("\\","/"),
            "val": str((out / 'images' / 'val').resolve()).replace("\\","/"),
            "test": str((out / 'images' / 'test').resolve()).replace("\\","/")
        },
        "labels": {
            "train": str((out / 'labels' / 'train').resolve()).replace("\\","/"),
            "val": str((out / 'labels' / 'val').resolve()).replace("\\","/"),
            "test": str((out / 'labels' / 'test').resolve()).replace("\\","/")
        },
        "affordances_file": str(cta_path.resolve()).replace("\\","/"),
        "crops_csv": "./affordance_crops/dataset.csv",
        "classnames": str(classes_txt_out.resolve()).replace("\\","/"),
        "num_affordances": len(affordances_all),
        "affordances_list": affordances_all,
        "notes": "Auto-generated; edit class_to_affordance.json to refine mappings."
    }
    print("Writing affordance_data.yaml to:", aff_yaml)
    aff_yaml.write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")

    print("\nSummary:")
    print("  classes_count:", len(classes_list))
    print("  affordances_detected:", affordances_all)
    if unknowns:
        print("\nWarning: the following classes were not found in the built-in mapping and were assigned ['grasp'] by default:")
        for u in unknowns:
            print("  -", u)
        print("Edit", cta_path, "to update affordances for those classes.\n")
    print("Done. If your dataset uses COCO/VOC style annotations instead of YOLO .txt, re-run with --classes_txt pointing to a classes file or provide a CSV of class names.")
    print("Next suggestions:")
    print("  1) If you want richer affordances per class, edit", cta_path)
    print("  2) Run your autocrop script to create affordance_crops/dataset.csv")
    print("  3) train: python backend/train_affordance.py --csv ./affordance_crops/dataset.csv --out ./backend_models/affordance_clf.pt --epochs 10 --batch 32")

if __name__ == "__main__":
    main()
