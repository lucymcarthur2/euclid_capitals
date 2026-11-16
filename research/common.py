"""Shared data loading and evaluation for the initial-capital extraction study.

Every model in this folder is scored by ``evaluate()`` on the same Euclid pages,
so differences in the results table come from the models, not the metric code.

Ground truth: ``Diss/annotations/32811_###.xml`` (Pascal VOC, 280 pages, 509 boxes),
in the pixel space of ``Diss/jpegs_cropped/32811_###.jpg`` (3250 x 4314).
The Diss folder is only read, never written. Set the DISS environment variable
if it lives somewhere else.
"""
from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DISS = Path(os.environ.get("DISS", r"data"))
EUCLID_IMG_DIR = DISS / "jpegs_cropped"
ORIGINAL_ANN_DIR = DISS / "annotations"
COMPILATIO_IMG_DIR = DISS / "compilatio" / "pages"
COMPILATIO_ANN_DIR = DISS / "compilatio_annotations"
ARTES_IMG_DIR = DISS / "artes_orandi" / "pages"
HERE = Path(__file__).resolve().parent
ARTES_ANN_DIR = HERE / "annotations_artes"  # built by make_book_labels.py + manual page check
PTOLEMY_IMG_DIR = DISS / "ptolemy" / "pages"
PTOLEMY_ANN_DIR = HERE / "annotations_ptolemy"  # extra TRAINING book (never used for testing)
# More extra TRAINING books (never used for testing): (image dir, annotation dir)
EXTRA_TRAIN_BOOKS = {
    "ptolemy": (PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR),
    "albumasar": (DISS / "albumasar" / "pages", HERE / "annotations_albumasar"),
    "arithmetica": (DISS / "arithmetica_boetij" / "pages", HERE / "annotations_arithmetica"),
}
# Corrected labels (make_corrected_labels.py): paragraph capitals only, 2 missing added, 1 chapter initial removed.
EUCLID_ANN_DIR = HERE / "annotations_corrected" if (HERE / "annotations_corrected").exists() else ORIGINAL_ANN_DIR
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)


@dataclass
class Page:
    page_id: str
    image_path: Path
    boxes: np.ndarray  # (N, 4) xmin, ymin, xmax, ymax in original pixels
    width: int
    height: int


def read_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    w = int(root.find("size/width").text)
    h = int(root.find("size/height").text)
    fname = root.find("filename").text
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        boxes.append([float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), w, h, fname


def load_pages(img_dir: Path = EUCLID_IMG_DIR, ann_dir: Path = EUCLID_ANN_DIR) -> list[Page]:
    """Pages that have both an annotation file and its image, sorted by page id."""
    pages = []
    for xml in sorted(ann_dir.glob("*.xml")):
        boxes, w, h, fname = read_voc(xml)
        img = img_dir / fname
        if not img.exists():
            raise FileNotFoundError(f"{xml.name} refers to missing image {img}")
        pages.append(Page(xml.stem, img, boxes, w, h))
    return pages


def gts_of(pages) -> dict:
    return {p.page_id: p.boxes for p in pages}


def kfold_ids(pages, k: int = 5) -> list[list[str]]:
    """Contiguous page-range folds (book order). Neighbouring pages share lighting, ink and
    often the same woodblocks, so random page splits would leak; contiguous blocks don't."""
    ids = [p.page_id for p in pages]
    return [list(a) for a in np.array_split(ids, k)]


def load_small(page: Page, long_side: int) -> tuple[np.ndarray, float]:
    """Greyscale page resized so its long side is `long_side`; cached as .npy. Returns (img, scale)."""
    from PIL import Image
    f = CACHE / f"{page.image_path.parent.parent.name}_{page.image_path.parent.name}_{page.page_id}_{long_side}.npy"
    s = long_side / max(page.width, page.height)
    if f.exists():
        return np.load(f), s
    im = Image.open(page.image_path).convert("L")
    im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    a = np.asarray(im)
    np.save(f, a)
    return a, s


# --------------------------------------------------------------------------- metrics

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_page(gt: np.ndarray, pred: np.ndarray, iou_thr: float):
    """Greedy one-to-one matching, highest-scoring prediction first (pred must be sorted).
    Returns (tp flag per prediction, IoUs of matched pairs)."""
    ious = iou_matrix(pred, gt)
    used = np.zeros(len(gt), bool)
    tp = np.zeros(len(pred), bool)
    matched = []
    for i in range(len(pred)):
        if len(gt) == 0:
            break
        cand = np.where(~used, ious[i], -1)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            used[j] = True; tp[i] = True; matched.append(float(cand[j]))
    return tp, matched


EMPTY = (np.zeros((0, 4), np.float32), np.zeros(0, np.float32))


def average_precision(gts: dict, preds: dict, iou_thr: float) -> float:
    """Single-class AP with 101-point interpolation (the COCO definition)."""
    n_gt = sum(len(g) for g in gts.values())
    scores, flags = [], []
    for pid, gt in gts.items():
        b, s = preds.get(pid, EMPTY)
        order = np.argsort(-s, kind="mergesort")
        tp, _ = match_page(gt, b[order], iou_thr)
        scores.extend(s[order]); flags.extend(tp)
    if n_gt == 0:
        return float("nan")
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    flags = np.asarray(flags, bool)[order]
    ctp = np.cumsum(flags); cfp = np.cumsum(~flags)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, rs, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def operating_point(gts: dict, preds: dict, score_thr: float, iou_thr: float = 0.5) -> dict:
    tp = fp = fn = 0
    ious, exact_pages, abs_count_err = [], 0, []
    for pid, gt in gts.items():
        b, s = preds.get(pid, EMPTY)
        keep = s >= score_thr
        b, s = b[keep], s[keep]
        order = np.argsort(-s, kind="mergesort")
        flags, m = match_page(gt, b[order], iou_thr)
        t = int(flags.sum()); f = len(flags) - t; n = len(gt) - t
        tp += t; fp += f; fn += n; ious.extend(m)
        exact_pages += int(f == 0 and n == 0)
        abs_count_err.append(abs(len(b) - len(gt)))
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(score_thr=float(score_thr), TP=tp, FP=fp, FN=fn, precision=p, recall=r,
                F1=2 * p * r / max(p + r, 1e-9),
                mean_IoU_matched=float(np.mean(ious)) if ious else float("nan"),
                pages_perfect=exact_pages / max(len(gts), 1),
                mean_abs_count_error=float(np.mean(abs_count_err)) if abs_count_err else float("nan"))


def best_threshold(gts: dict, preds: dict, iou_thr: float = 0.5) -> float:
    """Score cut-off maximising F1. Only ever call this on TRAINING pages."""
    s = [v[1] for k, v in preds.items() if k in gts]
    all_s = np.concatenate(s) if s else np.zeros(0)
    if len(all_s) == 0:
        return 0.5
    cands = np.unique(np.quantile(all_s, np.linspace(0, 1, 201)))
    return float(max(cands, key=lambda t: operating_point(gts, preds, t, iou_thr)["F1"]))


def evaluate(gts: dict, preds: dict, score_thr: float) -> dict:
    res = {
        "AP50": average_precision(gts, preds, 0.5),
        "AP75": average_precision(gts, preds, 0.75),
        "AP50_95": float(np.mean([average_precision(gts, preds, t) for t in np.arange(0.5, 0.951, 0.05)])),
    }
    res.update(operating_point(gts, preds, score_thr, 0.5))
    return res


def bootstrap_ci(gts: dict, preds_list: list, metric, n: int = 1000, seed: int = 0):
    """Page-level bootstrap. Returns per-model (lo, hi) and, for 2 models, the (lo, hi) of B - A."""
    rng = np.random.default_rng(seed)
    ids = list(gts)
    vals = [[] for _ in preds_list]
    for _ in range(n):
        pick = rng.choice(len(ids), len(ids), replace=True)
        g = {f"{k}": gts[ids[i]] for k, i in enumerate(pick)}
        for m, preds in enumerate(preds_list):
            p = {f"{k}": preds.get(ids[i], EMPTY) for k, i in enumerate(pick)}
            vals[m].append(metric(g, p))
    vals = np.asarray(vals)
    cis = [(float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for v in vals]
    diff = None
    if len(preds_list) == 2:
        d = vals[1] - vals[0]
        diff = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return cis, diff


# --------------------------------------------------------------------------- prediction io

def save_preds(preds: dict, path: Path, meta: dict | None = None):
    data = {"meta": meta or {}, "preds": {k: {"boxes": np.asarray(b).tolist(), "scores": np.asarray(s).tolist()}
                                          for k, (b, s) in preds.items()}}
    Path(path).write_text(json.dumps(data))


def load_preds(path: Path):
    data = json.loads(Path(path).read_text())
    preds = {k: (np.asarray(v["boxes"], np.float32).reshape(-1, 4), np.asarray(v["scores"], np.float32))
             for k, v in data["preds"].items()}
    return preds, data["meta"]
