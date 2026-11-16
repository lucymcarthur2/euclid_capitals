"""Run a trained detector over every page image of a book and save all detections (score >= 0.01)
as outputs/preds_<tag>_<book>.json, for scoring with evaluate_book.py.

usage: python predict_book.py BOOK [--weights weights/mbv3_all_pages.pth] [--tag mbv3_all]
  BOOK: compilatio | artes | euclid | a folder of page images
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import DISS, OUT, HERE, save_preds
from train_detector import build_model, LONG

BOOKS = {"compilatio": DISS / "compilatio" / "pages", "artes": DISS / "artes_orandi" / "pages",
         "euclid": DISS / "jpegs_cropped", "ptolemy": DISS / "ptolemy" / "pages"}

ap = argparse.ArgumentParser()
ap.add_argument("book")
ap.add_argument("--weights", type=Path, default=HERE / "weights" / "mbv3_all_pages.pth")
ap.add_argument("--tag", default="mbv3_all")
ap.add_argument("--pre", choices=["none", "grey", "ink"], default="none",
                help="test-time preprocessing: grey = channel mean; ink = brightest channel per pixel, "
                     "which keeps black printed ink but fades hand-applied colour (paint is bright in "
                     "at least one channel)")
a = ap.parse_args()


def preprocess(x):  # x: HxWx3 float in [0, 1]
    if a.pre == "grey":
        x = np.repeat(x.mean(2, keepdims=True), 3, 2)
    elif a.pre == "ink":
        x = np.repeat(x.max(2, keepdims=True), 3, 2)
    return np.ascontiguousarray(x, dtype=np.float32)
folder = BOOKS.get(a.book, Path(a.book))
name = a.book if a.book in BOOKS else folder.name

torch.set_num_threads(4)
model = build_model(pretrained=False)
model.load_state_dict(torch.load(a.weights, map_location="cpu"))
model.eval()

files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
               and not p.name.startswith("extracted_object"))
preds, times = {}, []
for i, f in enumerate(files):
    im = Image.open(f).convert("RGB")
    s = LONG / max(im.size)
    small = preprocess(np.asarray(im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR), np.float32) / 255)
    t0 = time.perf_counter()
    with torch.no_grad():
        o = model([torch.from_numpy(small).permute(2, 0, 1)])[0]
    times.append(time.perf_counter() - t0)
    preds[f.stem] = (o["boxes"].numpy() / s, o["scores"].numpy())
    if i % 25 == 0:
        print(f"{i + 1}/{len(files)} {f.name}", flush=True)
suffix = "" if a.pre == "none" else f"_{a.pre}"
save_preds(preds, OUT / f"preds_{a.tag}{suffix}_{name}.json",
           dict(model=a.tag, book=name, pre=a.pre, weights=str(a.weights), sec_per_page=float(np.mean(times))))
print(f"saved {len(preds)} pages, {np.mean(times):.2f}s/page")
