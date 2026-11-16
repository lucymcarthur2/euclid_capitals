"""Model B - "no deep learning" reference: image-processing proposals (proposals.py) scored by a
gradient-boosted classifier on hand-crafted features. Same 5 page-block folds as the detector.

usage: python classical_model.py
"""
import time

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import OUT, load_pages, kfold_ids, iou_matrix, save_preds
from proposals import propose, binarise
from train_detector import load_rgb

def hog(patch: np.ndarray, cells: int = 4, bins: int = 9) -> np.ndarray:
    """Minimal HOG: 32x32 patch, 4x4 cells, 9 unsigned orientation bins, L2-normalised."""
    p = cv2.resize(patch, (32, 32)).astype(np.float32)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0); gy = cv2.Sobel(p, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy); ang = (np.arctan2(gy, gx) % np.pi) / np.pi * bins
    idx = np.minimum(ang.astype(int), bins - 1)
    c = 32 // cells
    h = np.zeros((cells, cells, bins), np.float32)
    for i in range(cells):
        for j in range(cells):
            h[i, j] = np.bincount(idx[i*c:(i+1)*c, j*c:(j+1)*c].ravel(),
                                  mag[i*c:(i+1)*c, j*c:(j+1)*c].ravel(), bins)
    h = h.ravel()
    return h / (np.linalg.norm(h) + 1e-6)


def features(rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = binarise(g)
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    edges = cv2.Canny(g, 60, 150) > 0
    H, W = g.shape
    out = []
    for x1, y1, x2, y2 in boxes.astype(int):
        w, h = x2 - x1, y2 - y1
        pi, pg = ink[y1:y2, x1:x2], g[y1:y2, x1:x2]
        cx1, cy1 = x1 + w // 5, y1 + h // 5
        inner = ink[cy1:y2 - h // 5, cx1:x2 - w // 5]
        r = 3
        ring = np.concatenate([pi[:r].ravel(), pi[-r:].ravel(), pi[:, :r].ravel(), pi[:, -r:].ravel()])
        right = ink[y1:y2, x2:min(x2 + w, W)]
        left = ink[y1:y2, max(x1 - w, 0):x1]
        n_cc = cv2.connectedComponents(pi.astype(np.uint8))[0] - 1
        out.append(np.concatenate([[
            w / W, h / H, w / h, pi.mean(), inner.mean() if inner.size else 0, ring.mean(),
            pg.mean() / 255, pg.std() / 255, edges[y1:y2, x1:x2].mean(),
            n_cc / (w * h) * 100, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
            sat[y1:y2, x1:x2].mean() / 255,
            right.mean() if right.size else 0, left.mean() if left.size else 0], hog(pg)]))
    return np.asarray(out, np.float32).reshape(len(boxes), 15 + 4 * 4 * 9)


def nms(b, s, thr=0.3):
    order = np.argsort(-s)
    keep = []
    for i in order:
        if not keep or iou_matrix(b[i:i + 1], b[keep]).max() <= thr:
            keep.append(i)
    return b[keep], s[keep]


if __name__ == "__main__":
    pages = load_pages()
    data, t_feat = {}, []
    for p in pages:
        rgb, s = load_rgb(p.image_path)
        t0 = time.perf_counter()
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        c = propose(g)
        X = features(rgb, c)
        t_feat.append(time.perf_counter() - t0)
        cand = c / s
        y = (iou_matrix(cand, p.boxes).max(1) >= 0.5) if len(p.boxes) and len(cand) else np.zeros(len(cand), bool)
        data[p.page_id] = (cand, X, y)
    print(f"features done, {np.mean(t_feat):.2f}s/page")

    preds = {}
    for k, test_ids in enumerate(kfold_ids(pages)):
        tr = [pid for pid in data if pid not in test_ids]
        X = np.concatenate([data[i][1] for i in tr]); y = np.concatenate([data[i][2] for i in tr])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
        clf.fit(X, y)
        for pid in test_ids:
            cand, Xt, _ = data[pid]
            sc = clf.predict_proba(Xt)[:, 1] if len(cand) else np.zeros(0)
            preds[pid] = nms(cand, sc) if len(cand) else (cand, sc)
        print(f"fold {k}: train pos={int(y.sum())} neg={int((~y).sum())}")
    save_preds(preds, OUT / "preds_classical_cv.json",
               dict(model="classical_proposals_gbm", sec_per_page=float(np.mean(t_feat))))
    print("saved")

    # Cross-book test: train on ALL Euclid pages, predict every page of the other two books
    from common import COMPILATIO_IMG_DIR, ARTES_IMG_DIR
    X = np.concatenate([d[1] for d in data.values()]); y = np.concatenate([d[2] for d in data.values()])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X, y)
    for book, folder in (("compilatio", COMPILATIO_IMG_DIR), ("artes", ARTES_IMG_DIR)):
        out = {}
        for f in sorted(folder.glob("*.jpg")):
            rgb, s = load_rgb(f)
            c = propose(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            if len(c) == 0:
                out[f.stem] = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32)); continue
            out[f.stem] = nms(c / s, clf.predict_proba(features(rgb, c))[:, 1])
        save_preds(out, OUT / f"preds_classical_all_{book}.json", dict(model="classical_proposals_gbm", book=book))
        print("saved", book)
