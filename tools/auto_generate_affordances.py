#!/usr/bin/env python3
"""
Auto-generate class_to_affordance.json and affordance_data.yaml.
Heuristics mapping common object names -> affordances (bottle, chair, knife, pen, etc).
If dataset contains classes.txt or YOLO label files, those class names will be used.
"""

import argparse
import json
from pathlib import Path
import re

DEFAULT_AFFORDANCES = [
    "grasp", "contain", "pour", "support", "sit", "push", "pull", "press", "cut", "write", "open", "hold"
]

# heuristic object->affordances (extendable)
HEURISTIC_MAP = {
    "bottle": ["grasp","contain","pour"],
    "cup": ["grasp","contain","pour"],
    "mug": ["grasp","contain","pour"],
    "chair": ["support","sit"],
    "sofa": ["support","sit"],
    "bench": ["support","sit"],
    "pen": ["grasp","write"],
    "pencil": ["grasp","write"],
    "knife": ["grasp","cut"],
    "fork": ["grasp","support","contain"],
    "spoon": ["grasp","contain"],
    "door": ["push","pull","open"],
    "handle": ["grasp","pull","push"],
    "phone": ["grasp","press","hold"],
    "book": ["grasp","contain"],
    "box": ["contain","support","grasp"],
    "bowl": ["contain","support"],
    "plate": ["support","contain"],
    "table": ["support"],
    "keyboard": ["press","grasp"],
    "monitor": ["hold","support"]
}

def read_classes_from_classes_txt(root: Path):
    candidates = list(root.glob("**/classes*.txt"))
    if not candidates:
        return []
    # take first
    txt = candidates[0]
    with txt.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    return lines

def read_classes_from_yolo_labels(root: Path):
    # look for label files, extract class ids then attempt to get classes.txt sibling
    labels = list(root.glob("**/*.txt"))
    if not labels:
        return []
    # try to locate classes.txt in parent of labels or repo root
    parent = root
    for p in [root] + list(root.parents):
        ptxt = p / "classes.txt"
        if ptxt.exists():
            return [ln.strip() for ln in ptxt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # fallback: class ids only -> return numeric class_0, class_1...
    ids = set()
    for lab in labels:
        for ln in lab.read_text(encoding="utf-8").splitlines():
            parts = ln.split()
            if parts:
                ids.add(parts[0])
    ids = sorted(list(ids))
    return [f"class_{i}" for i in ids]

def heuristic_map_name(name: str):
    name_l = name.lower()
    # try direct matches
    for k in HEURISTIC_MAP:
        if k in name_l:
            return HEURISTIC_MAP[k]
    # try splitting on non-alphanum and check tokens
    tokens = re.split(r"[^a-zA-Z0-9]+", name_l)
    for t in tokens:
        if t in HEURISTIC_MAP:
            return HEURISTIC_MAP[t]
    # fallback: generic grasp/support
    return ["grasp","support"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="IIT_AAF dataset root (images/annotations)")
    p.add_argument("--out", required=True, help="output folder (affordance_dataset)")
    p.add_argument("--classes_txt", default=None, help="optional classes.txt to use")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    classes = []
    if args.classes_txt:
        classes = [ln.strip() for ln in Path(args.classes_txt).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        classes = read_classes_from_classes_txt(src)
        if not classes:
            classes = read_classes_from_yolo_labels(src)
    classes = list(dict.fromkeys(classes))  # unique preserve order

    if not classes:
        print("No classes discovered via classes.txt or YOLO labels. Will attempt to infer from folder/file names.")
        # look for any images and try to parse parent folder names
        images = list(src.glob("**/*.jpg")) + list(src.glob("**/*.png")) + list(src.glob("**/*.jpeg"))
        names = set()
        for im in images[:500]:
            names.add(im.parent.name)
        classes = sorted(list(names))[:50]

    print(f"Detected classes: {len(classes)} -> {classes[:20]}")

    # build class -> affordances
    mapping = {}
    affordances_set = set()
    for c in classes:
        affs = heuristic_map_name(c)
        mapping[c] = affs
        for a in affs:
            affordances_set.add(a)

    # ensure affordance list contains defaults
    for a in DEFAULT_AFFORDANCES:
        affordances_set.add(a)

    affordances = sorted(list(affordances_set))

    # write files
    class_to_affordance = mapping
    (out / "class_to_affordance.json").write_text(json.dumps(class_to_affordance, indent=2))
    data_yaml = {
        "train": "affordance_dataset/images/train",
        "val": "affordance_dataset/images/val",
        "test": "affordance_dataset/images/test",
        "nc": len(classes),
        "names": classes,
        "affordances": affordances
    }
    (out / "affordance_data.yaml").write_text(json.dumps(data_yaml, indent=2))
    if not (out / "classes.txt").exists():
        (out / "classes.txt").write_text("\n".join(classes))
    print("Wrote class_to_affordance.json and affordance_data.yaml to:", out)

if __name__ == "__main__":
    main()
