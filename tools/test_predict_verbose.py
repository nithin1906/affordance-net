from ultralytics import YOLO
import torch, os
from pathlib import Path

SRC = "affordance_crops/crops/00_00000090_0.jpg"
MODEL = "backend/models/best.pt"
OUT_DIR = Path("runs/detect/debug_predict")

print("\n=== RUNNING DEBUG DETECTION TEST ===")
print("Image:", SRC)
print("Model path:", MODEL)
print("-------------------------------------\n")

# Load model
m = YOLO(MODEL)
print("Model names:", getattr(m, "names", None))
print("Model object type:", type(m))
print("-------------------------------------\n")

# Run prediction at very low threshold
print("Running predict(conf=0.01, imgsz=640, save=False)...\n")
res = m.predict(source=SRC, conf=0.01, imgsz=640, save=False, verbose=True, augment=False)

print("\nReturned result type:", type(res), " | results count:", len(res))

if len(res) == 0:
    print("No results returned from model.predict()")
else:
    r = res[0]
    boxes = getattr(r, "boxes", None)
    print("Boxes object:", boxes)

    if boxes is None:
        print("boxes is None - model returned no detections")
    else:
        try:
            print("xyxy:", boxes.xyxy.tolist())
            print("conf:", boxes.conf.tolist())
            print("cls :", boxes.cls.tolist())
        except Exception as e:
            print("Error accessing box tensors:", e)

print("-------------------------------------\n")

# Run and save visualization
print("Running predict(save=True) to generate a visual output image...\n")
OUT_DIR.mkdir(parents=True, exist_ok=True)
res2 = m.predict(source=SRC, conf=0.01, imgsz=640, save=True, verbose=True)
print("\nSaved prediction results, check:  runs/detect/")
print("\n-------------------------------------")

# Inspect the model checkpoint file
print("\nInspecting checkpoint file content keys...\n")
try:
    ckpt = torch.load(MODEL, map_location="cpu")
    print("Checkpoint type:", type(ckpt))
    if isinstance(ckpt, dict):
        print("Checkpoint keys:", list(ckpt.keys()))
except Exception as e:
    print("Error reading checkpoint:", e)

print("\nDone.\n")