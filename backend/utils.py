# backend/utils.py
import io
from PIL import Image
import numpy as np

def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")

def pil_to_np(p: Image.Image) -> np.ndarray:
    return np.array(p)[:, :, ::-1]  # RGB to BGR for OpenCV/Ultralytics if needed

def crop_bbox_np(np_img, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = np_img.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    return np_img[y1:y2, x1:x2, :]

def crop_bbox_pil(pil_img, bbox):
    x1, y1, x2, y2 = [int(x) for x in bbox]
    return pil_img.crop((x1, y1, x2, y2))
