# backend/models.py
import os
import io
import logging
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics import YOLO

logger = logging.getLogger("affordance.models")
logger.setLevel(logging.INFO)

# Paths
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
AFFORDANCE_PATH = os.path.join(MODELS_DIR, "affordance_clf.pt")
DETECTOR_PATH = os.path.join(MODELS_DIR, "best.pt")  # your custom detector (optional)


class AffordanceClassifier:
    """
    Loads a PyTorch multi-label classifier saved as a state_dict or full model.
    Accepts PIL.Image crops and returns a vector of affordance probabilities.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = None
        self.classes = None
        self._load_or_dummy()

    def _load_or_dummy(self):
        if os.path.exists(AFFORDANCE_PATH):
            logger.info("Loading affordance classifier from %s", AFFORDANCE_PATH)
            checkpoint = torch.load(AFFORDANCE_PATH, map_location=self.device)
            # assume checkpoint is a dict {'state_dict':..., 'classes': [...] } OR raw model
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                # build a small MobileNetV2 head (same as train script)
                model = torch.hub.load('pytorch/vision', 'mobilenet_v2', weights=None)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(checkpoint["classes"]))
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                self.model = model.to(self.device).eval()
                self.classes = checkpoint["classes"]
                logger.info("Affordance classifier loaded with classes: %s", self.classes)
            elif isinstance(checkpoint, torch.nn.Module):
                self.model = checkpoint.to(self.device).eval()
                self.classes = getattr(checkpoint, "classes", None)
                logger.info("Loaded affordance model (module) classes=%s", self.classes)
            else:
                # legacy: assume full model state
                try:
                    model = torch.hub.load('pytorch/vision', 'mobilenet_v2', weights=None)
                    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
                    model.load_state_dict(checkpoint)
                    self.model = model.to(self.device).eval()
                    self.classes = [f"aff{i}" for i in range(model.classifier[1].out_features)]
                    logger.info("Loaded classifier from legacy checkpoint, classes auto-generated")
                except Exception as e:
                    logger.exception("Failed to load checkpoint: %s", e)
                    self._set_dummy()
        else:
            logger.warning("Affordance classifier not found at %s — using dummy classifier.", AFFORDANCE_PATH)
            self._set_dummy()

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _set_dummy(self):
        self.model = None
        self.classes = ["grasp", "pour", "cut", "sit_on", "support", "write", "hold", "open", "push", "pull"]  # example
        logger.info("Using DUMMY classifier returning zeros for classes: %s", self.classes)

    def predict(self, pil_img: Image.Image) -> List[Tuple[str, float]]:
        """
        Returns list of (class, prob) sorted descending
        """
        if self.model is None:
            # dummy: zeros
            return [(c, 0.0) for c in self.classes]

        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)  # assume raw logits multi-label
            if out.shape[1] == len(self.classes):
                probs = torch.sigmoid(out).cpu().numpy()[0]
            else:
                # if single-class softmax
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        return list(zip(self.classes, probs.tolist()))


class Detector:
    def __init__(self, device="cpu"):
        self.device = device
        # Try custom detector first; otherwise fallback to 'yolov8n.pt' which ultralytics will fetch
        model_path = DETECTOR_PATH if os.path.exists(DETECTOR_PATH) else "yolov8n.pt"
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        self.model = YOLO(model_path)
        logger.info("Loaded detector: %s", model_path)

    def detect(self, np_img: np.ndarray):
        """
        Runs YOLO detector on an OpenCV / numpy image (BGR).
        Returns list of detections: each dict has bbox (x1,y1,x2,y2), conf, cls, label.
        """
        # ultralytics expects RGB images
        rgb = np_img[:, :, ::-1]
        results = self.model(rgb, imgsz=640, conf=0.25)[0]
        detections = []
        for box, score, cls in zip(results.boxes.xyxy.cpu().numpy(),
                                   results.boxes.conf.cpu().numpy(),
                                   results.boxes.cls.cpu().numpy()):
            x1, y1, x2, y2 = map(float, box)
            label = self.model.names[int(cls)] if self.model.names else str(int(cls))
            detections.append({"bbox": [x1, y1, x2, y2], "conf": float(score), "class_id": int(cls), "label": label})
        return detections
