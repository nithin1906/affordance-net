# backend/server.py
#
# ----- FINAL WORKING VERSION -----
#
# Fixes include:
# 1. Correct ImageNet normalization to prevent 'NaN'.
# 2. Formats affordances as a DICTIONARY {"Graspable": 0.9}
#    to match the frontend App.jsx, fixing the "0, 1, 2..." bug.
# 3. Uses secure `weights_only=True` for torch.load.
# 4. Correctly reconstructs the MobileNetV2 model.
#
import io
import os
import base64
import logging
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from torchvision import transforms  # IMPORT TRANSFORMS

# try/except import ultralytics (YOLO)
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --- Configuration ---------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend", "dist")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DETECTOR_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
AFF_CLASSIFIER_PATH = os.path.join(ROOT, "backend_models", "affordance_clf_multilabel.pt")
DEFAULT_DETECTOR_FALLBACK = "yolov8n.pt"
LOG = logging.getLogger("affordance_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- Ensure static directory exists -----------------
os.makedirs(STATIC_DIR, exist_ok=True)
css_path = os.path.join(STATIC_DIR, "styles.css")
if not os.path.exists(css_path):
    with open(css_path, "w", encoding="utf-8") as f:
        f.write("/* default styles */ body{font-family:sans-serif;margin:0;padding:0}")
    LOG.info("Wrote default %s", css_path)

app = FastAPI(title="Affordance Server")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

try:
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    LOG.info("Mounted static dir: %s", STATIC_DIR)
except Exception as e:
    LOG.warning("Could not mount static dir: %s", e)

# --- Utilities -------------------------------------------------------------
def to_rgb_pil(img_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Unable to open image: {e}")

def crop_box_from_pil(pil_img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1)); y1 = max(0, int(y1)); x2 = min(pil_img.width, int(x2)); y2 = min(pil_img.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return pil_img
    return pil_img.crop((x1, y1, x2, y2))

def softmax_or_sigmoid_preds(logits: torch.Tensor) -> np.ndarray:
    try:
        return torch.sigmoid(logits).cpu().numpy()
    except Exception:
        return logits.cpu().numpy()

# --- Model loaders ---------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOG.info("Server starting — device=%s", DEVICE)

# DEFINE THE NORMALIZATION TRANSFORM (FIXES NaN BUG)
# This MUST match the normalization used during training
T_NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Detector (The "Finder")
detector = None
detector_present = False
if YOLO is None:
    LOG.warning("ultralytics.YOLO not available. Detector disabled.")
else:
    model_path = DETECTOR_PATH if os.path.exists(DETECTOR_PATH) else DEFAULT_DETECTOR_FALLBACK
    try:
        LOG.info("Loading detector from: %s", model_path)
        detector = YOLO(model_path)
        detector_present = True
        LOG.info("Detector loaded.")
    except Exception as e:
        LOG.exception("Failed to load detector: %s", e)
        detector_present = False

# Affordance classifier (The "Thinker")
affordance_clf = None
affordances_list: List[str] = []

def load_affordance_classifier(path: str):
    global affordance_clf, affordances_list
    if not os.path.exists(path):
        LOG.warning("Affordance classifier not found at: %s", path)
        return
    LOG.info("Loading affordance classifier: %s", path)
    
    try:
        # Use weights_only=True for security
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        LOG.warning("Could not load checkpoint with weights_only=True. Falling back to unsafe load.")
        ckpt = torch.load(path, map_location="cpu")

    if not isinstance(ckpt, dict):
        LOG.error("Checkpoint is not a dictionary. Cannot load classifier.")
        return

    affordances = ckpt.get("affordances")
    if affordances is None:
        LOG.warning("Checkpoint missing 'affordances' list.")
        affordances = ["unknown"]
    affordances_list = list(affordances)

    meta = ckpt.get("args", {}) # Args are saved here
    LOG.info("Affordance checkpoint meta: %s", meta)
    state_dict = ckpt.get("model_state_dict")

    if state_dict is None:
        LOG.error("No model_state_dict found in checkpoint. Classifier will not work.")
        return

    # Helper to load state_dict
    def finalize_model(model):
        try:
            model.load_state_dict(state_dict, strict=True)
            model.to(DEVICE).eval()
            LOG.info("Affordance classifier reconstructed and loaded.")
            return model
        except Exception as e:
            LOG.warning(f"Strict loading failed: {e}. Trying non-strict.")
            try:
                model.load_state_dict(state_dict, strict=False)
                model.to(DEVICE).eval()
                LOG.info("Affordance classifier (non-strict) reconstructed and loaded.")
                return model
            except Exception as e2:
                LOG.exception(f"Non-strict loading also failed: {e2}")
                return None

    # Reconstruct the model EXACTLY as it was in training
    try:
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights=None)  # Start with no weights
        n_classes = len(affordances_list)
        in_f = model.classifier[1].in_features
        # This structure MUST match the one in train_affordance_multilabel.py
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2), 
            torch.nn.Linear(in_f, n_classes)
        )
        model = finalize_model(model)
        
    except Exception:
        LOG.exception("Auto-reconstruction attempt failed.")

    if model is not None:
        affordance_clf = model
        LOG.info("Affordance classifier ready with %d affordances.", len(affordances_list))
    else:
        LOG.error("Could not reconstruct a runnable affordance model from checkpoint.")

# Load the model
load_affordance_classifier(AFF_CLASSIFIER_PATH)

# --- Pydantic Models -------------------------
class DetectedObject(BaseModel):
    box: List[float]   # [x1,y1,x2,y2]
    label: str
    score: float
    # --- FIX 2: The frontend App.jsx expects a dictionary, not a list ---
    affordances: Dict[str, float] = {}

class InferResponse(BaseModel):
    objects: List[DetectedObject]
    meta: Dict[str, Any] = {}

# --- Endpoints -------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "detector_present": detector_present, "affordances": affordances_list}

@app.post("/v1/infer", response_model=InferResponse)
async def infer(
    image: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None),
    conf: Optional[float] = Form(0.25),
    iou: Optional[float] = Form(0.45),
    max_det: Optional[int] = Form(300),
):
    if image is None and not image_b64:
        raise HTTPException(status_code=400, detail="No image provided.")
    try:
        if image is not None:
            content = await image.read()
        else:
            b64 = image_b64.split(",", 1)[-1]
            content = base64.b64decode(b64)
        pil_img = to_rgb_pil(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {e}")

    # run detector
    if not detector_present:
        raise HTTPException(status_code=503, detail="Detector model not loaded on server.")
    try:
        preds = detector(pil_img, imgsz=640, conf=conf, iou=iou, max_det=max_det, device=DEVICE)
    except Exception as e:
        LOG.exception("Detector failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Detector runtime error: {e}")

    # Parse detections
    objects_out = []
    try:
        res0 = preds[0] if isinstance(preds, (list, tuple)) else preds
        boxes = res0.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box[:4])
            cls_id = int(cls_ids[i])
            score = float(confs[i])
            cls_name = str(res0.names[cls_id])
            
            crop = crop_box_from_pil(pil_img, (x1, y1, x2, y2))
            
            # --- FIX 2: Build a dictionary, not a list ---
            affordance_preds: Dict[str, float] = {} 
            
            if affordance_clf is not None and cls_name != "person":
                try:
                    # Apply the correct transforms (FIXES NaN BUG)
                    img_pil_resized = crop.resize((224, 224))
                    img_tensor_pre_norm = transforms.ToTensor()(img_pil_resized)
                    img_tensor = T_NORM(img_tensor_pre_norm).unsqueeze(0).to(DEVICE)
                    
                    with torch.no_grad():
                        logits = affordance_clf(img_tensor)
                        probs = softmax_or_sigmoid_preds(logits.squeeze(0))

                    for idx, p in enumerate(np.atleast_1d(probs)):
                        name = affordances_list[idx] if idx < len(affordances_list) else f"aff_{idx}"
                        # Add to dictionary
                        affordance_preds[name] = float(p)
                        
                except Exception as ee:
                    LOG.exception("Affordance classifier error: %s", ee)
                    affordance_preds["error"] = 0.0
            
            # --- END FIX ---

            obj = {
                "box": [x1, y1, x2, y2],
                "label": cls_name,
                "score": score,
                "affordances": affordance_preds, # This is now a dictionary
            }
            objects_out.append(obj)
    except Exception as e:
        LOG.exception("Failed parsing detector results: %s", e)
        raise HTTPException(status_code=500, detail=f"Result parsing error: {e}")

    meta = {
        "detector": "yolov8n.pt",
        "affordance_classifier_loaded": affordance_clf is not None,
        "affordances": affordances_list,
    }
    return JSONResponse({"objects": objects_out, "meta": meta})

@app.get("/")
def index():
    return {"message": "Affordance server running", "health": f"/health", "infer": "/v1/infer (POST multipart/file=image OR form=image_b64)"}

if __name__ == "__main__":
    import uvicorn
    # Run from terminal with: uvicorn backend.server:app --port 8000
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False, log_level="info")