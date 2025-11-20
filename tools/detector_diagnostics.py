# tools/detector_diagnostics.py
r"""
Detector diagnostics script.

Saves visualizations and prints detection summaries for multiple conf thresholds and image sizes.

Usage:
  python tools/detector_diagnostics.py --imgs IMAGE1 [IMAGE2 ...]
  python tools/detector_diagnostics.py           # will auto-find up to 8 sample images in common folders

Outputs:
  - runs/detect/diagnostics/<conf>_<imgsz>/  -> saved visual images
  - printed table of results (image, conf, imgsz, #detections, top confidences)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import json
import sys

MODEL_PATH = Path("backend/models/best.pt")  # adjust if needed
SEARCH_PATHS = [
    Path("affordance_crops/crops"),
    Path("affordance_dataset/images"),
    Path("affordance_dataset/images/train"),
    Path("runs/train"),
    Path("runs/detect"),
]

def find_images(limit=8):
    imgs = []
    for p in SEARCH_PATHS:
        if p.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                imgs.extend(sorted(p.glob(ext)))
        if len(imgs) >= limit:
            break
    return imgs[:limit]

def run_diag(model, img_paths, confs=(0.5,0.3,0.15,0.05), imgszs=(640,960)):
    results_summary = []
    out_root = Path("runs/detect/diagnostics")
    out_root.mkdir(parents=True, exist_ok=True)

    for conf in confs:
        for imgsz in imgszs:
            # save visualizations per config
            project = out_root / f"conf{conf}_imgsz{imgsz}"
            project.mkdir(parents=True, exist_ok=True)
            print(f"\n=== conf={conf} imgsz={imgsz} -> saving to {project} ===")
            # run predictions with save=True to write overlays
            # note: model.predict can accept list of sources
            res = model.predict(source=[str(p) for p in img_paths], conf=conf, imgsz=imgsz, save=True, project=str(project), name="vis", exist_ok=True)

            # summarize results returned
            # res is list of Result objects (one per image)
            for i, r in enumerate(res):
                imgp = Path(img_paths[i])
                boxes = getattr(r, "boxes", None)
                dets = []
                if boxes is not None:
                    for b in boxes:
                        try:
                            cls = int(b.cls.item())
                            confb = float(b.conf.item())
                            xyxy = b.xyxy.tolist()
                        except Exception:
                            # some ultralytics versions use dict-like b
                            cls = None; confb = None; xyxy = None
                        dets.append((cls, confb, xyxy))
                dets_sorted = sorted(dets, key=lambda x: (x[1] is not None, x[1]), reverse=True)
                top_confs = [round(x[1],4) for x in dets_sorted[:5] if x[1] is not None]
                print(f"{imgp.name} — detections: {len(dets)} top_confs: {top_confs}")
                results_summary.append({
                    "image": str(imgp),
                    "conf": conf,
                    "imgsz": imgsz,
                    "n_dets": len(dets),
                    "top_confs": top_confs
                })
    return results_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs", nargs="*", help="image paths")
    parser.add_argument("--model", default=str(MODEL_PATH), help="detector path")
    parser.add_argument("--confs", nargs="*", type=float, default=[0.5,0.3,0.15,0.05], help="confidence thresholds")
    parser.add_argument("--imgszs", nargs="*", type=int, default=[640,960], help="image sizes")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(2)

    model = YOLO(str(model_path))
    print("Loaded model. class names:", model.names)

    if args.imgs:
        imgs = [Path(x) for x in args.imgs]
        for p in imgs:
            if not p.exists():
                print("ERROR: image not found:", p)
                sys.exit(3)
    else:
        imgs = find_images(limit=8)
        if not imgs:
            print("No sample images found. Provide --imgs paths.")
            sys.exit(4)

    print(f"Running diagnostics on {len(imgs)} images: {[p.name for p in imgs]}")
    summary = run_diag(model, imgs, confs=args.confs, imgszs=args.imgszs)

    # write summary file
    out = Path("runs/detect/diagnostics/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out}")

if __name__ == "__main__":
    main()
