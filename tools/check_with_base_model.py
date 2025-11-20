# tools/check_with_base_model.py
from ultralytics import YOLO
src = "affordance_crops/crops/00_00000090_0.jpg"
print("Using image:", src)
m = YOLO("yolov8n.pt")   # base pretrained from Ultralytics
print("Base model names:", m.names)
res = m.predict(source=src, conf=0.01, imgsz=640, save=True, verbose=True)
print("Result count:", len(res))
if len(res)>0:
    r=res[0]; boxes=getattr(r,"boxes",None)
    print("boxes len:", len(boxes) if boxes is not None else "None")
