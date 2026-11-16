"""Score prediction files against labels made in the review tool (labels_review/<book>.json).

Ground truth for a page = draft boxes the reviewer kept + boxes the reviewer added. Only pages
marked 'checked' count. Reported at the Euclid-CV cut-off (weights/threshold.json) and at 0.5.

usage: python evaluate_reviewed.py BOOK PRED_FILE [PRED_FILE ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from common import HERE, OUT, evaluate, load_preds, EMPTY, operating_point


def reviewed_gts(book):
    rev = json.loads((HERE / "labels_review" / f"{book}.json").read_text())
    draft_f = HERE / "labels_draft" / f"{book}.json"
    draft = json.loads(draft_f.read_text()) if draft_f.exists() else {}
    gts = {}
    for pid, r in rev.items():
        if not r.get("checked"):
            continue
        kept = [b[:4] for b, ok in zip(draft.get(pid, []), r["verdicts"]) if ok]
        gts[pid] = np.array(kept + r["added"], dtype=float).reshape(-1, 4)
    return gts


if __name__ == "__main__":
    book = sys.argv[1]
    gts = reviewed_gts(book)
    thr = json.loads((HERE / "weights" / "threshold.json").read_text())["threshold"]
    print(f"{book}: {len(gts)} checked pages, {sum(len(v) for v in gts.values())} capitals")
    cols = ["AP50", "AP75", "precision", "recall", "F1", "TP", "FP", "FN", "mean_IoU_matched"]
    print("| predictions | cut-off | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 2))
    for f in sys.argv[2:]:
        preds, _ = load_preds(Path(f))
        for t in (thr, 0.5):
            r = evaluate(gts, preds, t)
            print(f"| {Path(f).stem} | {t:.2f} | " + " | ".join(
                f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
        errs = []
        for pid, gt in gts.items():
            b, s = preds.get(pid, EMPTY); k = s >= thr
            o = operating_point({pid: gt}, {pid: (b[k], s[k])}, -np.inf)
            if o["FP"] or o["FN"]:
                errs.append((pid, o["FN"], o["FP"]))
        print("   pages with errors at the Euclid cut-off (page, missed, false):", errs)
