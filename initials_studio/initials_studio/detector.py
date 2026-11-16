"""The capital detector: Faster R-CNN with a MobileNetV3-Large FPN backbone, fine-tuned on real pages of
Euclid (Ratdolt, 1482) plus three more Ratdolt-era books whose woodcut pictures act as hard negatives.
Runs on CPU at ~0.3-0.5 s per page."""
import json
import threading

import numpy as np
import torch
from PIL import Image

from .config import MODELS_DIR

LONG = 1024
_model, _lock = None, threading.Lock()


def model_info():
    f = MODELS_DIR / "model.json"
    return json.loads(f.read_text()) if f.exists() else {}


def get_model():
    global _model
    with _lock:
        if _model is None:
            from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            torch.set_num_threads(max(1, torch.get_num_threads()))
            m = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None, min_size=LONG, max_size=LONG,
                                                  box_detections_per_img=30, box_score_thresh=0.01)
            m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
            m.load_state_dict(torch.load(MODELS_DIR / model_info().get("weights", "capitals_mbv3_v2.pth"),
                                         map_location="cpu"))
            m.eval()
            _model = m
        return _model


@torch.no_grad()
def detect(image: Image.Image, floor: float):
    """-> list of (box [x1,y1,x2,y2] in image pixels, score) with score >= floor."""
    m = get_model()
    im = image.convert("RGB")
    s = LONG / max(im.size)
    small = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(small, np.float32) / 255).permute(2, 0, 1)
    o = m([t])[0]
    out = []
    for b, sc in zip(o["boxes"].numpy() / s, o["scores"].numpy()):
        if sc >= floor:
            out.append(([round(float(v), 1) for v in b], round(float(sc), 4)))
    return out
