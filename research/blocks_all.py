"""Same-block matching across ALL books at once (see block_matcher.py for the method).

1. Collect every initial: labelled boxes where a book has labels, reviewed boxes for Judgement, the
   detector's boxes (score >= MIN_SCORE) for the rest. False alarms can be removed in the Patterns page.
2. Cheap pre-check: every pair is compared on 24x24 blurred ink maps in all 8 orientations
   (one matrix product). Each initial keeps its K closest candidates (plus any pair above CHEAP_T).
   On Euclid this keeps ~98% of true same-block pairs while skipping ~85% of the work.
3. Full check on the kept pairs: ECC alignment in the cheap check's best orientation + ink-map
   correlation (block_matcher.pair_score's scoring, one orientation).
Output: cache/blocks_all.npz (sparse pair scores) -> block_report_all.py.

usage: python blocks_all.py
"""
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np

from common import (HERE, OUT, CACHE, DISS, load_pages, load_preds, COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR,
                    ARTES_IMG_DIR, ARTES_ANN_DIR, PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR)
from block_matcher import crop, prep, orient, ncc, S, B, CRIT

K, CHEAP_T, MIN_SCORE = 60, 0.45, 0.5

# book -> (printer, date as far as checked; "" = not checked)
BOOKS = {
    "euclid": ("Ratdolt", "1482"), "compilatio": ("Ratdolt", "1489"), "artes": ("Ratdolt", "1485"),
    "ptolemy": ("Ratdolt", ""), "judgement": ("Ratdolt", ""),
    "hyginus_blavis_1488": ("Thomas de Blavis", "1488"),
    "avienus_strata_1488": ("Antonius de Strata", "1488"),
    "aquinas_santritter_1488": ("Santritter & de Sanctis", "1488"),
    "jandun_santritter_1488": ("Santritter & de Sanctis", "1488"),
    "niger_santritter_1488": ("Santritter", ""),
    "tabulae_hamman_1492": ("Johannes Hamman", "1492"),
    "zacut_liechtenstein": ("Petrus Liechtenstein", ""),
}


def gt_items(book, img_dir, ann_dir):
    return [(book, p.page_id, p.image_path, [int(v) for v in b]) for p in load_pages(img_dir, ann_dir) for b in p.boxes]


def det_items(book, folder, skip=()):
    f = OUT / f"preds_mbv3_v2_{book}.json"
    if not f.exists():
        print("no detections yet for", book); return []
    preds, _ = load_preds(f)
    out, k = [], 0
    for pid in sorted(preds, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]):
        b, s = preds[pid]
        for x in b[s >= MIN_SCORE]:
            if k not in skip:
                out.append((book, pid, Path(folder) / f"{pid}.jpg", [int(v) for v in x]))
            k += 1
    return out


def judgement_items():
    draft = json.loads((HERE / "labels_draft" / "judgement_of_the_stars.json").read_text())
    rev = json.loads((HERE / "labels_review" / "judgement_of_the_stars.json").read_text())
    out = []
    for pid, r in rev.items():
        kept = [b for b, ok in zip(draft.get(pid, []), r["verdicts"]) if ok] + r.get("added", [])
        out += [("judgement", pid, HERE / "other_books" / "judgement_of_the_stars" / f"{pid}.jpg", [int(v) for v in b])
                for b in kept]
    return out


def hyginus_items():  # the reviewed tiles from render_detections (101: E inside a woodcut, 106: show-through)
    line = re.compile(r"^(\d+) (\S+) \[(\d+), (\d+), (\d+), (\d+)\]")
    out = []
    for ln in (OUT / "hyginus_dets.txt").read_text(encoding="utf-8-sig").splitlines():
        m = line.match(ln.strip())
        if m and int(m.group(1)) not in (101, 106):
            out.append(("hyginus_blavis_1488", m.group(2), HERE / "other_books" / "hyginus_blavis_1488" / f"{m.group(2)}.jpg",
                        [int(m.group(i)) for i in range(3, 7)]))
    return out


def all_items():
    items = gt_items("euclid", DISS / "jpegs_cropped", HERE / "annotations_corrected")
    items += gt_items("compilatio", COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR)
    items += gt_items("artes", ARTES_IMG_DIR, ARTES_ANN_DIR)
    items += gt_items("ptolemy", PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR)
    items += judgement_items() + hyginus_items()
    for b in ["avienus_strata_1488", "aquinas_santritter_1488", "jandun_santritter_1488", "niger_santritter_1488",
              "tabulae_hamman_1492", "zacut_liechtenstein"]:
        items += det_items(b, HERE / "other_books" / b)
    return items


def small(f, k):
    x = cv2.GaussianBlur(cv2.resize(f["m"], (24, 24), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    x = orient(x, k)[2:-2, 2:-2]
    x = x - x.mean()
    return (x / (np.linalg.norm(x) + 1e-9)).ravel()


_F = None


def _init(F):
    global _F
    _F = F
    cv2.setNumThreads(1)


def _prep(it):
    return prep(crop(it[2], it[3]))


def _full(a, b, o):
    ga, ma = orient(a["gf"], o), orient(a["m"], o)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(b["gf"], ga, M, cv2.MOTION_AFFINE, CRIT, None, 5)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)
    wa = cv2.warpAffine(ma, M, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return ncc(wa[B:-B, B:-B], b["m"][B:-B, B:-B])


def _pair(args):
    """full score of item j turned by orientation o, aligned onto item i. If the cheap check preferred a
    mirrored pose, the best plain rotation o2 is tried too and wins unless the mirror is clearly better
    (near-symmetric designs - O, H, I, N - look almost the same mirrored)."""
    i, j, o, o2 = args
    s = _full(_F[j], _F[i], o)
    if o >= 4:
        s2 = _full(_F[j], _F[i], o2)
        if s2 >= s - 0.05:
            return i, j, o2, s2
    return i, j, o, s


if __name__ == "__main__":
    from multiprocessing import Pool
    items = all_items()
    books = [it[0] for it in items]
    print({b: books.count(b) for b in dict.fromkeys(books)}, "total", len(items), flush=True)
    t0 = time.time()
    with Pool(4) as pool:
        F = pool.map(_prep, items, chunksize=16)
    print(f"prepared in {time.time()-t0:.0f}s", flush=True)
    V = np.stack([np.stack([small(f, k) for k in range(8)]) for f in F]).astype(np.float32)  # N,8,D
    U = V[:, 0, :]
    n = len(F)
    C = np.empty((n, n), np.float32); O = np.empty((n, n), np.int8); O2 = np.empty((n, n), np.int8)
    for s in range(0, n, 256):  # C[j, i] = best cheap score of j (turned) vs i upright; O = that turn
        P = np.einsum("nkd,md->nmk", V[s:s + 256], U)
        C[s:s + 256], O[s:s + 256], O2[s:s + 256] = P.max(2), P.argmax(2), P[:, :, :4].argmax(2)
    Cs = np.maximum(C, C.T); np.fill_diagonal(Cs, -9)
    top = np.argsort(-Cs, 1)[:, :K]
    keep = np.zeros((n, n), bool); keep[np.arange(n)[:, None], top] = True
    keep |= Cs >= CHEAP_T; keep |= keep.T
    I, J = np.where(np.triu(keep, 1))
    jobs = [(i, j, int(O[j, i]), int(O2[j, i])) for i, j in zip(I, J)]
    print(f"{len(jobs)} pairs to check fully ({100 * len(jobs) / (n * (n - 1) / 2):.1f}% of all)", flush=True)
    res = []
    t0 = time.time()
    with Pool(4, initializer=_init, initargs=(F,)) as pool:
        for k, r in enumerate(pool.imap_unordered(_pair, jobs, chunksize=64)):
            res.append(r)
            if k % 20000 == 0:
                print(f"{k}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
    r = np.array(res, dtype=np.float64)
    np.savez_compressed(CACHE / "blocks_all.npz", pi=r[:, 0].astype(np.int32), pj=r[:, 1].astype(np.int32),
                        po=r[:, 2].astype(np.int8), ps=r[:, 3].astype(np.float32), cheap_orient=O,
                        books=np.array(books), pages=np.array([it[1] for it in items]),
                        paths=np.array([str(it[2]) for it in items]), boxes=np.array([it[3] for it in items]),
                        book_info=json.dumps(BOOKS))
    print("saved", CACHE / "blocks_all.npz", f"{time.time()-t0:.0f}s")
