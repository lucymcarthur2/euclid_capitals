"""Stage 1 of the classical pipeline: cheap, high-recall candidate boxes for decorated capitals.

A woodcut initial is a near-square block of very dense ink (black ground, white tracery,
usually a printed frame). Body text, rubrics and diagrams are made of thin, separated strokes.
Two complementary cues give candidates:

  (a) connected ink components that are big and nearly square (the frame holds the block together);
  (b) blobs of high local ink density (catches worn blocks whose frame is broken).

Everything runs on a greyscale copy whose long side is LONG_SIDE pixels (~1/4 of the scan).
"""
from __future__ import annotations

import cv2
import numpy as np

LONG_SIDE = 1024


def binarise(g: np.ndarray) -> np.ndarray:
    """Ink = 1. Local (adaptive) threshold so uneven lighting and page curvature don't matter."""
    g = cv2.GaussianBlur(g, (3, 3), 0)
    ink = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 12)
    return ink


def _boxes_from_labels(n, stats):
    x, y, w, h, a = stats[1:n].T
    return np.stack([x, y, x + w, y + h], 1).astype(np.float32), a


def propose(g: np.ndarray, ink: np.ndarray | None = None,
            min_side: int = 28, max_side: int = 220, min_aspect: float = 0.5) -> np.ndarray:
    """Candidate boxes (x1, y1, x2, y2) in the coordinates of `g`."""
    if ink is None:
        ink = binarise(g)
    out = []

    # (a) connected components after a small closing (re-joins frames broken by wear)
    closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    b, _ = _boxes_from_labels(n, stats)
    out.append(b)

    # (b) dense-ink blobs at two scales
    for k, thr in ((15, 0.42), (25, 0.38)):
        dens = cv2.blur(ink.astype(np.float32), (k, k))
        m = (dens > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        b, _ = _boxes_from_labels(n, stats)
        # the box filter shrinks/grows blobs by ~k/2; snap each blob to the ink inside it
        snapped = []
        for x1, y1, x2, y2 in b.astype(int):
            pad = k // 2
            X1, Y1 = max(x1 - pad, 0), max(y1 - pad, 0)
            X2, Y2 = min(x2 + pad, ink.shape[1]), min(y2 + pad, ink.shape[0])
            ys, xs = np.nonzero(ink[Y1:Y2, X1:X2])
            if len(xs) == 0:
                continue
            snapped.append([X1 + xs.min(), Y1 + ys.min(), X1 + xs.max() + 1, Y1 + ys.max() + 1])
        out.append(np.asarray(snapped, np.float32).reshape(-1, 4))

    b = np.concatenate(out)
    w = b[:, 2] - b[:, 0]; h = b[:, 3] - b[:, 1]
    keep = (w >= min_side) & (h >= min_side) & (w <= max_side) & (h <= max_side) & \
           (np.minimum(w, h) / np.maximum(w, h) >= min_aspect)
    b = b[keep]
    return dedupe(b, 0.9)


def dedupe(b: np.ndarray, thr: float) -> np.ndarray:
    """Drop near-identical boxes (IoU > thr); order-independent, no scores needed."""
    from common import iou_matrix
    if len(b) == 0:
        return b
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = b[np.argsort(-area)]
    keep = []
    for i in range(len(b)):
        if not keep or iou_matrix(b[i:i + 1], b[keep]).max() <= thr:
            keep.append(i)
    return b[keep]
