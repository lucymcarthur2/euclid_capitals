"""Same-block matcher: were two initials printed from the SAME physical woodblock?

Two impressions of one block share every detail of the cut design (and, later, the same cracks and
breaks). A recut copy shares the overall design but differs in the fine vine work. So:

  1. Crop every capital: Euclid (Ratdolt, Venice 1482) from the corrected labels, the other book from
     the detector's boxes (minus the false alarms listed in SKIP).
  2. For a pair, find SIFT keypoints, match them, and fit a similarity transform with RANSAC
     (rotation + scale + shift: impressions differ only in position/scale, not shape).
  3. Warp one crop onto the other and measure how much of the cut DESIGN lines up: the outline of
     the inked areas, compared with a 2-pixel tolerance in both directions (symmetric chamfer
     agreement, 0..1). Ink spread and wear blur this a little; a different hand-cut block cannot
     line up across the whole design.
  4. Calibrate on Euclid alone: Ratdolt reused each block many times inside the book, so the
     within-Euclid nearest-neighbour scores show what 'same block' looks like.

usage: python block_matcher.py OTHER_BOOK_DIR DETS_TXT NAME [--skip 101,106]
   e.g. python block_matcher.py other_books/hyginus_blavis_1488 outputs/hyginus_dets.txt hyginus --skip 101,106
Outputs: cache/blocks_<NAME>.npz (all pair scores) and outputs/blocks_<NAME>_*.jpg / .csv.
"""
import argparse
import re
import time
from pathlib import Path

import cv2
import numpy as np

from common import CACHE, OUT, load_pages

S = 160  # every initial is squared to S x S after cropping tightly to its black frame
B = 8    # border ignored when scoring (frame edges and crop slop)


def tight(grey):
    """Crop to the initial's own frame: the bounding box of the largest connected ink region
    (frame + ground + design are one dark component; neighbouring text letters are small)."""
    ink = cv2.threshold(cv2.GaussianBlur(grey, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return grey
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    if w * h < 0.3 * grey.size:  # frame not found reliably: keep the detector box
        return grey
    return grey[y:y + h, x:x + w]


def prep(grey):
    """grey crop -> squared, contrast-normalised image + blurred ink map used for scoring."""
    g = cv2.resize(tight(grey), (S, S), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    ink = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    m = cv2.GaussianBlur(ink.astype(np.float32) / 255, (0, 0), 1.2)  # tolerate ink spread / wear
    return dict(g=g, m=m, gf=cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2),
                s=cv2.resize(m, (40, 40), interpolation=cv2.INTER_AREA)[3:-3, 3:-3])


def ncc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)


# The 8 ways a square image can be turned. A block can be printed turned 90/180/270 degrees (your
# dissertation found upside-down blocks, and a V used as an upside-down A), but an impression can
# never be MIRRORED: a mirror-image match means a recut reversed copy, not the same block.
ORIENT = ["0", "rot90", "rot180", "rot270", "mirror", "mirror+rot90", "mirror+rot180", "mirror+rot270"]


def orient(x, k):
    if k >= 4:
        x = x[:, ::-1]
    return np.ascontiguousarray(np.rot90(x, -(k % 4)))


def _ecc_score(ga, ma, b):
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(b["gf"], ga, M, cv2.MOTION_AFFINE, CRIT, None, 5)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)  # no convergence: score the identity alignment
    wa = cv2.warpAffine(ma, M, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return ncc(wa[B:-B, B:-B], b["m"][B:-B, B:-B])


def pair_score(a, b):
    """Best orientation of a, aligned onto b (affine ECC from identity, both are frame-squared), scored
    by the normalised correlation of their ink maps inside the border.
    Returns (score, orientation index). ~1 = same cut design, ~0 = unrelated.
    A cheap 40x40 correlation picks the most likely turned orientation; the full alignment is run
    for the upright pose and for that orientation only."""
    best_k = int(np.argmax([ncc(orient(a["s"], k), b["s"]) for k in range(1, 8)])) + 1
    s0 = _ecc_score(a["gf"], a["m"], b)
    sk = _ecc_score(orient(a["gf"], best_k), orient(a["m"], best_k), b)
    return (s0, 0) if s0 >= sk else (sk, best_k)


def crop(img_path, box, pad=0.02):
    im = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    x1, y1, x2, y2 = box
    p = pad * max(x2 - x1, y2 - y1)
    x1, y1 = max(int(x1 - p), 0), max(int(y1 - p), 0)
    x2, y2 = min(int(x2 + p), im.shape[1]), min(int(y2 + p), im.shape[0])
    return im[y1:y2, x1:x2]


_F, _NE = None, 0


def _init(feats, n_euclid):
    global _F, _NE
    _F, _NE = feats, n_euclid
    cv2.setNumThreads(1)


def _row(i):
    """scores of crop i against every later crop (other-book vs other-book pairs skipped)."""
    return i, [(j, *pair_score(_F[j], _F[i])) for j in range(i + 1, len(_F)) if i < _NE or j < _NE]


def euclid_items():
    items = []
    for p in load_pages():
        for i, b in enumerate(p.boxes):
            items.append((f"euclid/{p.page_id}#{i}", p.image_path, [int(v) for v in b]))
    return items


def other_items(folder, dets_txt, skip):
    line = re.compile(r"^(\d+) (\S+) \[(\d+), (\d+), (\d+), (\d+)\]")
    items = []
    for ln in Path(dets_txt).read_text(encoding="utf-8-sig").splitlines():
        m = line.match(ln.strip())
        if m and int(m.group(1)) not in skip:
            items.append((f"{Path(folder).name}/{m.group(2)}#{m.group(1)}", Path(folder) / f"{m.group(2)}.jpg",
                          [int(m.group(i)) for i in range(3, 7)]))
    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("dets"); ap.add_argument("name")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    skip = {int(x) for x in a.skip.split(",") if x}
    E, O = euclid_items(), other_items(a.folder, a.dets, skip)
    t0 = time.time()
    feats = [prep(crop(p, b)) for _, p, b in E + O]
    print(f"{len(E)} Euclid + {len(O)} {a.name} crops prepared in {time.time()-t0:.0f}s", flush=True)
    n = len(feats)
    score = np.full((n, n), np.nan, np.float32)
    turn = np.zeros((n, n), np.int8)  # orientation of crop j (row) that best fits crop i; see ORIENT
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(4, initializer=_init, initargs=(feats, len(E))) as pool:
        for k, (i, row) in enumerate(pool.imap_unordered(_row, range(n), chunksize=4)):
            for j, s, o in row:
                score[i, j] = score[j, i] = s
                turn[j, i] = o  # j turned by ORIENT[o] matches i
            if k % 50 == 0:
                print(f"{k}/{n} rows, {time.time()-t0:.0f}s", flush=True)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / f"blocks_{a.name}.npz", score=score, turn=turn, n_euclid=len(E),
                        ids=np.array([x[0] for x in E + O]), paths=np.array([str(x[1]) for x in E + O]),
                        boxes=np.array([x[2] for x in E + O]))
    print("saved", CACHE / f"blocks_{a.name}.npz", f"{time.time()-t0:.0f}s")
