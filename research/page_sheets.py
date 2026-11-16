"""Render every labelled page of a book (8 per sheet) with its boxes, for a full visual label check.

usage: python page_sheets.py [euclid|compilatio]
Optional extra boxes from a prediction file are drawn in blue: python page_sheets.py artes outputs/preds_x.json
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from common import (OUT, load_pages, EUCLID_IMG_DIR, EUCLID_ANN_DIR, COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR,
                    ARTES_IMG_DIR, ARTES_ANN_DIR, PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR, EXTRA_TRAIN_BOOKS, load_preds)
from train_detector import load_rgb

book = sys.argv[1] if len(sys.argv) > 1 else "euclid"
pages = {"euclid": lambda: load_pages(EUCLID_IMG_DIR, EUCLID_ANN_DIR),
         "compilatio": lambda: load_pages(COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR),
         "artes": lambda: load_pages(ARTES_IMG_DIR, ARTES_ANN_DIR),
         "ptolemy": lambda: load_pages(PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR),
         "albumasar": lambda: load_pages(*EXTRA_TRAIN_BOOKS["albumasar"]),
         "arithmetica": lambda: load_pages(*EXTRA_TRAIN_BOOKS["arithmetica"])}[book]()
pages.sort(key=lambda p: int("".join(c for c in p.page_id if c.isdigit()) or 0))
extra = load_preds(Path(sys.argv[2]))[0] if len(sys.argv) > 2 else {}
font = ImageFont.load_default(size=26)
TW, TH = 400, 531
d = OUT / f"page_sheets_{book}"; d.mkdir(exist_ok=True)
for s in range(0, len(pages), 8):
    sheet = Image.new("RGB", (4 * TW, 2 * TH), "white")
    for j, p in enumerate(pages[s:s + 8]):
        a, sc = load_rgb(p.image_path)
        im = Image.fromarray(a).resize((TW, round(a.shape[0] * TW / a.shape[1])))
        k = TW / p.width
        dr = ImageDraw.Draw(im)
        for b in p.boxes:
            dr.rectangle([v * k for v in b], outline=(255, 0, 0), width=3)
        for b in extra.get(p.page_id, ([], []))[0]:
            dr.rectangle([v * k for v in b], outline=(0, 120, 255), width=2)
        dr.rectangle([0, 0, 150, 30], fill="yellow"); dr.text((4, 2), p.page_id.replace("page_", "p"), fill="black", font=font)
        sheet.paste(im.crop((0, 0, TW, TH)), ((j % 4) * TW, (j // 4) * TH))
    sheet.save(d / f"sheet_{s // 8:02d}.jpg", quality=80)
print("sheets:", (len(pages) + 7) // 8, "->", d)
