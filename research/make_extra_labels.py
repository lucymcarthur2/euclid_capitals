"""Labels for two extra TRAINING books (never used for testing): Albumasar and Boethius' Arithmetica.

Neither had sorted crops, so every detection the v1 model made at score >= 0.3
(outputs/<book>_dets.txt, tiles in outputs/<book>_dets_*.jpg) was judged by eye as
'woodcut paragraph capital' or not. KEEP lists the tile numbers judged to be capitals; all
other tiles (zodiac roundels, sun faces, animals, plain printed initials, number grids)
become hard negatives simply by being unlabelled. Every page was then checked on contact
sheets for capitals the model missed entirely; those are added in ADD.

usage: python make_extra_labels.py
"""
import re
import xml.etree.ElementTree as ET

import cv2

from common import DISS, HERE, OUT
from make_book_labels import write_voc

BOOKS = {
    "albumasar": dict(pages=DISS / "albumasar" / "pages",
                      keep={0, 1, 6, 7, 9, 10, 13, 14, 32, 33, 34}, add={}),
    "arithmetica": dict(pages=DISS / "arithmetica_boetij" / "pages",
                        keep={0, 1, 2, 8, 10, 14, 15, 16},
                        # missed by the v1 detector, found by the classical second opinion (second_opinion.py)
                        add={"page_24": [(36, 304, 168, 460)], "page_30": [(306, 452, 412, 572)],
                             "page_94": [(40, 624, 180, 771)]}),
}

line = re.compile(r"^(\d+) (\S+) \[(\d+), (\d+), (\d+), (\d+)\] ([\d.]+)$")
for book, cfg in BOOKS.items():
    boxes = {}
    for ln in (OUT / f"{book}_dets.txt").read_text(encoding="utf-8-sig").splitlines():
        m = line.match(ln.strip())
        if m and int(m.group(1)) in cfg["keep"]:
            boxes.setdefault(m.group(2), []).append(tuple(int(m.group(i)) for i in range(3, 7)))
    for pid, extra in cfg["add"].items():
        boxes.setdefault(pid, []).extend(extra)
    out = HERE / f"annotations_{book}"
    out.mkdir(exist_ok=True)
    n = 0
    for p in sorted(cfg["pages"].glob("*.jpg")):
        h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape
        bs = boxes.get(p.stem, [])
        write_voc(out / f"{p.stem}.xml", p.name, w, h, bs)
        n += len(bs)
    print(f"{book}: {n} capitals on {len(list(out.glob('*.xml')))} pages -> {out}")
