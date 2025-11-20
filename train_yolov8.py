# train_yolov8.py
# Usage:
#   python train_yolov8.py --data ./affordance_dataset/data.yaml --epochs 50 --batch 8 --device 0
#
# Requires: ultralytics (YOLOv8), torch (with CUDA)
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to data.yaml (YOLO format)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--device', default='0', help="cuda device id or 'cpu' (e.g. 0 or 'cpu')")
    p.add_argument('--model', default='yolov8n.pt', help='base model to fine-tune (yolov8n.pt recommended)')
    p.add_argument('--save', default='runs/train/affordance', help='where to save runs')
    args = p.parse_args()

    # Convert device param (if user passes numeric device as int via CLI)
    device = args.device
    if device.isdigit():
        device = int(device)

    print(f"Training YOLOv8 on data: {args.data}")
    print(f"Using device: {device}, base model: {args.model}")

    model = YOLO(args.model)
    model.train(data=args.data,
                epochs=args.epochs,
                batch=args.batch,
                device=device,
                project=Path(args.save).parent,
                name=Path(args.save).name,
                imgsz=640,  # 640 is a good default for real-time/object detection
                workers=4)

    print("Training finished. Best weights usually at runs/train/<name>/weights/best.pt")

if __name__ == '__main__':
    main()
