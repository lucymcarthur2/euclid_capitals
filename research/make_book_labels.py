"""Build Pascal VOC ground truth for a book from the dissertation's hand-sorted capital crops,
WITHOUT using any model trained here.

The crops in Diss/<book>/organised/<letter>/extracted_object_page_N.jpg_k.png are exact pixel
cut-outs of Diss/<book>/pages/page_N.jpg, so template matching recovers each crop's box.
Every page is then checked by eye on contact sheets (page_sheets.py <book>); disagreements are
fixed in ADD / REMOVE below.

usage: python make_book_labels.py artes|ptolemy
Output: annotations_<book>/page_N.xml
"""
import re
import sys
import xml.etree.ElementTree as ET

import cv2

from common import DISS, HERE

FOLDER = {"artes": "artes_orandi", "ptolemy": "ptolemy"}
# Fixes from the visual page check: {book: {page_stem: [(x1, y1, x2, y2), ...]}}
ADD: dict = {"artes": {}, "ptolemy": {
    "page_3": [(1750, 2383, 2014, 2676)],                            # 'I' missing from the sorted crops
    "page_4": [(1784, 2097, 2075, 2406), (976, 2693, 1266, 3047)],   # 'S' and 'N' missing from the sorted crops
}}
REMOVE: dict = {"artes": {}, "ptolemy": {}}   # boxes whose centre falls inside any listed rectangle are dropped


def locate(page_grey, crop_grey):
    r = cv2.matchTemplate(page_grey, crop_grey, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(r)
    h, w = crop_grey.shape
    return (x, y, x + w, y + h), score


def write_voc(path, fname, w, h, boxes):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = fname
    size = ET.SubElement(root, "size")
    for k, v in (("width", w), ("height", h), ("depth", 3)):
        ET.SubElement(size, k).text = str(v)
    for b in boxes:
        o = ET.SubElement(root, "object"); ET.SubElement(o, "name").text = "object"
        bb = ET.SubElement(o, "bndbox")
        for k, v in zip(("xmin", "ymin", "xmax", "ymax"), b):
            ET.SubElement(bb, k).text = str(int(v))
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def inside(b, r):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


if __name__ == "__main__":
    book = sys.argv[1]
    src = DISS / FOLDER[book]
    out = HERE / f"annotations_{book}"
    out.mkdir(exist_ok=True)
    pat = re.compile(r"extracted_object_(page_\d+)\.jpg_(\d+)\.png")
    crops = [f for f in (src / "organised").rglob("*.png") if pat.fullmatch(f.name)]
    boxes, low = {}, []
    for f in sorted(crops):
        page = pat.fullmatch(f.name).group(1)
        pg = cv2.imread(str(src / "pages" / f"{page}.jpg"), cv2.IMREAD_GRAYSCALE)
        b, s = locate(pg, cv2.imread(str(f), cv2.IMREAD_GRAYSCALE))
        if s < 0.9:
            low.append((f.name, round(s, 3)))
        boxes.setdefault(page, []).append(b)
    print(f"{len(crops)} crops; low-confidence matches: {low}")
    n = 0
    for p in sorted((src / "pages").glob("*.jpg")):
        bs = [b for b in boxes.get(p.stem, []) if not any(inside(b, r) for r in REMOVE[book].get(p.stem, []))]
        bs += ADD[book].get(p.stem, [])
        h, w = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).shape
        write_voc(out / f"{p.stem}.xml", p.name, w, h, bs)
        n += len(bs)
    print(f"wrote {len(list(out.glob('*.xml')))} pages, {n} boxes -> {out}")
