"""Run the dissertation's Faster R-CNN (ResNet-50 FPN, trained on synthetic pages) on every
labelled page of a book and save ALL detections so AP can be computed properly.

Model set-up is copied from Diss/train_faster_r_cnn.py and Diss/run_RPN.py: torchvision
fasterrcnn_resnet50_fpn, 2 classes, ToTensor only (no extra normalisation), default resize
(min 800 / max 1333).

The Diss folder holds several candidate weight files; compare.py scores each on Euclid and keeps
the strongest, so the original model is judged at its best.

usage: python run_baseline.py TAG [euclid|compilatio|artes]
  TAG is one of the keys of CANDIDATES (output: outputs/preds_baseline_<TAG>[_<book>].json)
"""
import sys
import time

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from common import (DISS, OUT, load_pages, save_preds, COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR,
                    ARTES_IMG_DIR, ARTES_ANN_DIR)

CANDIDATES = {
    "b4_models": DISS / "models" / "model_lr_0.005_momentum_0.9_epochs_10_batch_4.pth",
    "b4_root": DISS / "model_lr_0.005_momentum_0.9_epochs_10_batch_4.pth",
    "model_weights": DISS / "model_weights.pth",
    "extract_capitals": DISS / "model_extract_capitals.pth",
}
tag = sys.argv[1] if len(sys.argv) > 1 else "b4_models"
book = sys.argv[2] if len(sys.argv) > 2 else "euclid"
torch.set_num_threads(4)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                                             box_score_thresh=0.01)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
model.load_state_dict(torch.load(CANDIDATES[tag], map_location="cpu"))
model.eval()

pages = {"euclid": lambda: load_pages(),
         "compilatio": lambda: load_pages(COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR),
         "artes": lambda: load_pages(ARTES_IMG_DIR, ARTES_ANN_DIR)}[book]()
preds, times = {}, []
for i, p in enumerate(pages):
    img = F.to_tensor(Image.open(p.image_path).convert("RGB"))
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model([img])[0]
    times.append(time.perf_counter() - t0)
    keep = out["labels"] == 1
    preds[p.page_id] = (out["boxes"][keep].numpy(), out["scores"][keep].numpy())
    if i % 20 == 0:
        print(f"{tag} {book} {i+1}/{len(pages)} {p.page_id} n={int(keep.sum())} {times[-1]:.1f}s", flush=True)

meta = dict(model="baseline_fasterrcnn_r50fpn_synthetic", weights=str(CANDIDATES[tag]), book=book,
            sec_per_page=float(np.mean(times)), params=sum(p.numel() for p in model.parameters()))
name = f"preds_baseline_{tag}.json" if book == "euclid" else f"preds_baseline_{tag}_{book}.json"
save_preds(preds, OUT / name, meta)
print("done", meta)
