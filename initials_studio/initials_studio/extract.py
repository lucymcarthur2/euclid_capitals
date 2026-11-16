"""Detection over a book, crop saving, and the 'does this look dodgy?' flags."""
import numpy as np
from PIL import Image

from .config import CROP_LONG_SIDE, DETECT_FLOOR
from . import store
from .detector import detect


def crop_stats(grey: np.ndarray) -> dict:
    """Cheap appearance statistics used by the flags."""
    import cv2
    g = cv2.GaussianBlur(grey, (5, 5), 0)
    ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # a woodcut initial is a dark block: its large ink pieces (the frame and ground may be broken
    # into several) together span most of the crop. Text, pictures and show-through don't.
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    frame = False
    big = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= 0.01 * grey.size]
    if big:
        x1 = min(st[k, 0] for k in big); y1 = min(st[k, 1] for k in big)
        x2 = max(st[k, 0] + st[k, 2] for k in big); y2 = max(st[k, 1] + st[k, 3] for k in big)
        dark = sum(st[k, cv2.CC_STAT_AREA] for k in big) / grey.size
        frame = bool((x2 - x1) >= 0.7 * grey.shape[1] and (y2 - y1) >= 0.7 * grey.shape[0] and dark >= 0.25)
    p10, p90 = np.percentile(grey, [10, 90])
    return dict(frame=frame, contrast=round(float(p90 - p10), 1), ink=round(float((ink > 0).mean()), 3))


def save_crop(book_id, cap_id, page_img: Image.Image, box):
    x1, y1, x2, y2 = box
    p = 0.04 * max(x2 - x1, y2 - y1)
    c = page_img.crop((max(int(x1 - p), 0), max(int(y1 - p), 0), min(int(x2 + p), page_img.width),
                       min(int(y2 + p), page_img.height))).convert("RGB")
    c.thumbnail((CROP_LONG_SIDE, CROP_LONG_SIDE), Image.LANCZOS)
    path = store.crop_path(book_id, cap_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    c.save(path, quality=92)
    return crop_stats(np.asarray(c.convert("L")))


def run_detection(book, progress=lambda f, msg: None, cancel=lambda: False):
    caps = [c for c in store.get_capitals(book["id"]) if c["status"] == "added"]  # keep hand-drawn boxes
    pages = book["pages"]
    for i, pg in enumerate(pages):
        if cancel():
            break
        with Image.open(store.page_path(book["id"], pg["id"])) as im:
            im.load()
            for k, (box, score) in enumerate(detect(im, DETECT_FLOOR)):
                cid = f"{pg['id'].lower()}_{k}"
                caps.append(dict(id=cid, page=pg["id"], box=box, score=score, status="auto",
                                 stats=save_crop(book["id"], cid, im, box)))
        progress((i + 1) / len(pages), f"Finding capitals: page {i + 1} of {len(pages)}")
    store.save_capitals(book["id"], caps)
    book["status"] = "detected"
    store.save_book(book)
    return caps


def add_manual(book_id, page_id, box):
    caps = store.get_capitals(book_id)
    n = sum(1 for c in caps if c["page"] == page_id and c["status"] == "added")
    cid = f"{page_id.lower()}_m{n}"
    with Image.open(store.page_path(book_id, page_id)) as im:
        im.load()
        stats = save_crop(book_id, cid, im, box)
    caps.append(dict(id=cid, page=page_id, box=[round(float(v), 1) for v in box], score=1.0, status="added",
                     stats=stats))
    store.save_capitals(book_id, caps)
    return cid


def visible(cap, s) -> bool:
    """Is this capital part of the book's results under the current settings?"""
    if cap["status"] == "rejected":
        return False
    return cap["status"] in ("kept", "added") or cap["score"] >= s["score_cutoff"]


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


FLAG_TEXT = {
    "low_confidence": "Low confidence",
    "size": "Unusual size for this book",
    "shape": "Not square like a woodcut initial",
    "no_frame": "No clear woodcut frame",
    "faint": "Faint: show-through or damage?",
    "overlap": "Overlaps another box",
    "edge": "Touches the page edge",
    "unclear_block": "Unclear block match",
    "one_of_a_kind": "Matches no other capital",
    "mirror": "Mirror image of another block: recut copy?",
}


def flag_book(book, caps, s, group_flags=None):
    """Attach a 'flags' list to every capital (computed fresh, so parameter changes apply instantly)."""
    vis = [c for c in caps if visible(c, s)]
    areas = np.array([(c["box"][2] - c["box"][0]) * (c["box"][3] - c["box"][1]) for c in vis]) if vis else np.ones(1)
    med_area = float(np.median(areas)) if len(areas) else 1.0
    contrasts = [c.get("stats", {}).get("contrast", 0) for c in vis]
    med_con = float(np.median(contrasts)) if contrasts else 0
    pages = {p["id"]: p for p in book.get("pages", [])}
    by_page = {}
    for c in vis:
        by_page.setdefault(c["page"], []).append(c)
    for c in caps:
        f = []
        x1, y1, x2, y2 = c["box"]
        w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        if c["status"] == "auto" and c["score"] < s["flag_low_confidence"]:
            f.append("low_confidence")
        a = w * h
        if len(vis) >= 5 and (a > s["flag_size_factor"] * med_area or a < med_area / s["flag_size_factor"]):
            f.append("size")
        if not 0.6 <= w / h <= 1.65:
            f.append("shape")
        st = c.get("stats", {})
        if st and not st.get("frame", True):
            f.append("no_frame")
        if st and med_con and st.get("contrast", med_con) < 0.55 * med_con:
            f.append("faint")
        if any(o is not c and iou(c["box"], o["box"]) > 0.2 for o in by_page.get(c["page"], [])):
            f.append("overlap")
        pg = pages.get(c["page"])
        if pg and pg.get("w"):  # crops-only example books have no page size
            m = 0.004 * max(pg["w"], pg["h"])
            if x1 <= m or y1 <= m or x2 >= pg["w"] - m or y2 >= pg["h"] - m:
                f.append("edge")
        if group_flags and c["id"] in group_flags:
            f += group_flags[c["id"]]
        c["flags"] = f
        c["visible"] = visible(c, s)
    return caps
