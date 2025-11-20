# train_yolo.py
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # or "yolov8s.pt"
model.train(data="affordance_dataset/affordance_data.yaml",
            epochs=100,
            imgsz=640,
            batch=16,
            device=0,           # or "0,1"
            project="runs/train",
            name="affordance_yolov8n_aug")
