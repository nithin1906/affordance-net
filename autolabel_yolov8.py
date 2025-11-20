# autolabel_yolov8.py
"""
Auto-label images using YOLOv8 (Ultralytics).
Creates YOLO-format .txt label files + placeholder affordance JSONs.

Usage:
  python autolabel_yolov8.py ^
    --images_dir ".\affordance_dataset\images\train" ^
    --labels_dir ".\affordance_dataset\labels\train" ^
    --aff_dir ".\affordance_dataset\affordances\train" ^
    --model yolov8n.pt ^
    --conf 0.35 ^
    --max_images 100
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image

try:
    from ultralytics import YOLO
except Exception:
    raise SystemExit("❌ Install ultralytics: pip install ultralytics pillow tqdm")

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bx = (x1 + x2) / 2.0
    by = (y1 + y2) / 2.0
    bw = x2 - x1
    bh = y2 - y1
    return bx / w, by / h, bw / w, bh / h

def write_yolo_label(label_path, detections, w, h):
    lines = []
    for d in detections:
        cx, cy, nw, nh = xyxy_to_yolo(*d['xyxy'], w, h)
        lines.append(f"{d['class_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} {d['conf']:.3f}")
    label_path.write_text("\n".join(lines))

def create_placeholder_affordance(aff_path, img_name, detections):
    data = {
        "image": img_name,
        "objects": [],
        "note": "Auto-generated placeholder affordances. Edit later."
    }
    for d in detections:
        data["objects"].append({
            "box": [int(d["xyxy"][0]), int(d["xyxy"][1]), int(d["xyxy"][2]), int(d["xyxy"][3])],
            "class": d["class_name"],
            "score": float(d["conf"]),
            "affordances": [],
            "parts": [],
            "anchors": []
        })
    aff_path.write_text(json.dumps(data, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True)
    p.add_argument("--labels_dir", required=True)
    p.add_argument("--aff_dir", required=True)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_images", type=int, default=-1)
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    aff_dir = Path(args.aff_dir)
    ensure_dir(labels_dir)
    ensure_dir(aff_dir)

    print(f"🚀 Loading model: {args.model}")
    model = YOLO(args.model)

    imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    if args.max_images > 0:
        imgs = imgs[:args.max_images]

    for img_path in tqdm(imgs, desc="Auto-labeling"):
        im = Image.open(img_path)
        w, h = im.size

        results = model.predict(source=str(img_path), conf=args.conf, imgsz=640, verbose=False, device=args.device)
        detections = []
        if results:
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls in zip(boxes, confs, classes):
                detections.append({
                    "class_id": int(cls),
                    "class_name": model.names[int(cls)],
                    "xyxy": box,
                    "conf": float(conf)
                })

        label_path = labels_dir / f"{img_path.stem}.txt"
        write_yolo_label(label_path, detections, w, h)

        aff_path = aff_dir / f"{img_path.stem}.json"
        create_placeholder_affordance(aff_path, img_path.name, detections)

    print("✅ Done! Labels saved to:", labels_dir)
    print("✅ Placeholder affordances saved to:", aff_dir)

if __name__ == "__main__":
    main()
