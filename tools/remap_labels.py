# tools/remap_labels.py
import json, os, shutil, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--labels_dir", default="affordance_dataset/labels", help="YOLO label files root")
parser.add_argument("--out_dir", default="affordance_dataset/labels_remapped", help="where to write remapped labels")
parser.add_argument("--classes_txt", default="affordance_dataset/classes.txt", help="output classes.txt")
parser.add_argument("--yaml_out", default="affordance_dataset/affordance_data.yaml", help="output data yaml")
parser.add_argument("--map_json", default=None, help="optional JSON file mapping original_id->new_id")
args = parser.parse_args()

# EDIT THIS: mapping of original class ids (as ints) -> new class index (0..K-1)
# Example: only keep class ids 56->0 (bottle), 0->1 (chair), 12->2 (cup) etc.
# You must fill this map according to your dataset's original class indexing.
# If map_json is provided, it overrides this dictionary.
class_map = {
    # ORIGINAL_ID: NEW_ID
    # e.g. 56 is original bottle in your label sample -> remap to new index 0 (bottle)
    56: 0,  # bottle
    0: 1,   # maybe 'chair' in your data (example)
    # add other mappings for cup, pen, knife, spoon, plate, person as needed
    # 12:2,  34:3, ...
}

if args.map_json:
    with open(args.map_json, "r") as f:
        class_map = json.load(f)
        # ensure keys are ints
        class_map = {int(k): int(v) for k, v in class_map.items()}

# Define final classes list in order NEW_ID -> class name (string)
# EDIT this to match what each NEW_ID means
classes = [
    "bottle",   # new_id 0
    "chair",    # new_id 1
    # add more class names matching indices assigned in class_map
]

out_dir = Path(args.out_dir)
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

labels_root = Path(args.labels_dir)
total = 0
kept = 0
for txt in labels_root.rglob("*.txt"):
    rel = txt.relative_to(labels_root)
    out_file = out_dir / rel
    out_file.parent.mkdir(parents=True, exist_ok=True)
    new_lines = []
    for line in txt.read_text().splitlines():
        line=line.strip()
        if not line:
            continue
        parts = line.split()
        orig_cls = int(parts[0])
        if orig_cls in class_map:
            new_cls = class_map[orig_cls]
            new_lines.append(" ".join([str(new_cls)] + parts[1:]))
            kept += 1
        total += 1
    out_file.write_text("\n".join(new_lines))
print(f"Processed {total} labels lines, kept {kept} remapped lines -> {out_dir}")

# write classes.txt
Path(args.classes_txt).parent.mkdir(parents=True, exist_ok=True)
with open(args.classes_txt, "w") as f:
    f.write("\n".join(classes))
print("Wrote classes file:", args.classes_txt)

# write data yaml (YOLOv8 style)
data_yaml = {
    "path": str(Path(".").resolve()),   # repo root
    "train": str((Path("affordance_dataset/images/train")).as_posix()),
    "val":   str((Path("affordance_dataset/images/val")).as_posix()),
    "names": classes
}
import yaml
with open(args.yaml_out, "w") as f:
    yaml.safe_dump(data_yaml, f)
print("Wrote data yaml:", args.yaml_out)
