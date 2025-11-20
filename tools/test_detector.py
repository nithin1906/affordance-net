# tools/test_detector.py
r"""Quick detector test script.
Usage:
  python tools/test_detector.py <image_path>
If no image_path provided, the script will try to find a sample image in common folders:
  - affordance_crops/crops (generated crops)
  - affordance_dataset/images (original dataset)
  - runs/detect (last detection outputs)
"""

import sys
from pathlib import Path
import json
from ultralytics import YOLO

# default model path (adjust if you keep your detector somewhere else)
MODEL_PATH = Path("backend/models/best.pt")

# fallback image search paths (repo-root relative)
SEARCH_PATHS = [
    Path("affordance_crops/crops"),
    Path("affordance_dataset/images"),
    Path("affordance_dataset/images/train"),
    Path("runs/detect"),
    Path("affordance_crops"),
]

def find_sample_image():
    for p in SEARCH_PATHS:
        if p.exists() and p.is_dir():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                found = list(p.glob(ext))
                if found:
                    return found[0]
    return None

def pretty_print_results(results, names):
    out = []
    for idx, r in enumerate(results):
        # r.boxes may be present as a Boxes object; convert
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            print(f"{idx}: no boxes")
            continue
        for b in boxes:
            # ultralytics Boxes contains xyxy tensor, cls, conf
            xyxy = b.xyxy.tolist() if hasattr(b, "xyxy") else None
            cls = int(b.cls.item()) if hasattr(b, "cls") else None
            conf = float(b.conf.item()) if hasattr(b, "conf") else None
            label = names.get(cls, str(cls)) if isinstance(names, dict) else (names[cls] if names else str(cls))
            out.append({
                "class_id": cls,
                "label": label,
                "confidence": round(conf, 4),
                "xyxy": [round(v, 1) for v in xyxy] if xyxy else None
            })
    print(json.dumps(out, indent=2))
    return out

def main():
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if img_arg:
        img_path = Path(img_arg)
        if not img_path.exists():
            print(f"ERROR: Provided image path does not exist: {img_path}")
            sys.exit(2)
    else:
        sample = find_sample_image()
        if sample is None:
            print("No sample image found in fallback folders. Please pass an image path.")
            sys.exit(3)
        img_path = sample
        print(f"No image provided — using sample: {img_path}")

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}. Place your detector weights at that path.")
        sys.exit(4)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    print("Model class names:", model.names)
    print(f"Running detect on: {img_path}")
    # run with lower conf if you want more sensitivity
    results = model.predict(source=str(img_path), conf=0.25, imgsz=640, save=False)
    # pretty print results
    detections = pretty_print_results(results, model.names)

    # save a visualization image (optional)
    try:
        out_dir = Path("runs/detect/test_detector")
        out_dir.mkdir(parents=True, exist_ok=True)
        # use model.predict save=True to save image, but we already have results; call model.show?
        # simplest: run predict with save=True to write visualization
        model.predict(source=str(img_path), conf=0.25, imgsz=640, save=True, project=str(out_dir), name="vis")
        print(f"Visualization saved to: {out_dir}")
    except Exception as e:
        print("Warning: could not write visualization:", e)

if __name__ == "__main__":
    main()
