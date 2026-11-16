"""Render every detection >= MIN_SCORE from a prediction file as numbered tiles, to label an
unlabelled TRAINING book by eye (each tile is then marked capital / not capital).

usage: python render_detections.py PAGES_DIR PRED_FILE NAME [min_score]
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import OUT, load_preds

pages_dir, pred_file, name = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
min_score = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
preds, _ = load_preds(pred_file)
font = ImageFont.load_default(size=18)
tiles = []
for pid in sorted(preds, key=lambda k: int("".join(c for c in k if c.isdigit()) or 0)):
    b, s = preds[pid]
    keep = s >= min_score
    for x, sc in zip(b[keep], s[keep]):
        print(len(tiles), pid, [int(v) for v in x], round(float(sc), 2))
        pad = 60
        X1, Y1 = max(int(x[0]) - pad, 0), max(int(x[1]) - pad, 0)
        with Image.open(pages_dir / f"{pid}.jpg") as im:
            c = im.convert("RGB").crop((X1, Y1, int(x[2]) + pad, int(x[3]) + pad))
        ImageDraw.Draw(c).rectangle([x[0] - X1, x[1] - Y1, x[2] - X1, x[3] - Y1], outline=(0, 140, 255), width=4)
        c.thumbnail((160, 160))
        t = Image.new("RGB", (160, 182), "white"); t.paste(c, (0, 22))
        ImageDraw.Draw(t).text((2, 1), f"#{len(tiles)} {pid.replace('page_', 'p')} {sc:.2f}", fill="black", font=font)
        tiles.append(t)
for s0 in range(0, len(tiles), 60):
    sh = Image.new("RGB", (10 * 160, 6 * 182), "white")
    for i, t in enumerate(tiles[s0:s0 + 60]):
        sh.paste(t, ((i % 10) * 160, (i // 10) * 182))
    sh.save(OUT / f"{name}_dets_{s0 // 60}.jpg", quality=85)
print("tiles", len(tiles))
