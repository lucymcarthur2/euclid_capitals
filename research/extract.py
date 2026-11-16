"""Extract paragraph initial capitals from a folder of page scans.

usage:
  python extract.py PAGES_DIR OUT_DIR [--weights weights/mbv3_v2_all_pages.pth] [--threshold T]

Writes OUT_DIR/crops/<page>_<n>.png (full-resolution crops, top-to-bottom order) and
OUT_DIR/detections.csv (page, n, x1, y1, x2, y2, score). CPU only, ~1 s per page.
The default threshold is the cross-validated F1-optimal cut-off that compare.py
records in weights/threshold.json.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_detector import build_model, LONG

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("pages", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--weights", type=Path, default=HERE / "weights" / "mbv3_v2_all_pages.pth")
ap.add_argument("--threshold", type=float, default=None)
args = ap.parse_args()

cfg_file = HERE / "weights" / "threshold.json"
thr = args.threshold if args.threshold is not None else (
    json.loads(cfg_file.read_text())["threshold"] if cfg_file.exists() else 0.5)

torch.set_num_threads(4)
model = build_model(pretrained=False)
model.load_state_dict(torch.load(args.weights, map_location="cpu"))
model.eval()

(args.out / "crops").mkdir(parents=True, exist_ok=True)
files = sorted(p for p in args.pages.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
total = 0
with open(args.out / "detections.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["page", "n", "x1", "y1", "x2", "y2", "score"])
    for f in files:
        im = Image.open(f).convert("RGB")
        s = LONG / max(im.size)
        small = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        t = torch.from_numpy(np.asarray(small, np.float32) / 255).permute(2, 0, 1)
        with torch.no_grad():
            o = model([t])[0]
        keep = o["scores"].numpy() >= thr
        boxes = o["boxes"].numpy()[keep] / s
        scores = o["scores"].numpy()[keep]
        order = np.argsort(boxes[:, 1]) if len(boxes) else []
        for n, i in enumerate(order):
            b = boxes[i].round().astype(int)
            im.crop(tuple(b)).save(args.out / "crops" / f"{f.stem}_{n}.png")
            w.writerow([f.name, n, *b, round(float(scores[i]), 4)])
        total += len(order)
        print(f"{f.name}: {len(order)} capitals")
print(f"{total} capitals from {len(files)} pages at threshold {thr:.2f}")
