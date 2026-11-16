"""Cross-book test: models trained ONLY on Euclid, scored on Compilatio and Artes Orandi.

Neither book was used for training, tuning or threshold choice. The operating threshold is the one
compare.py picked from Euclid cross-validation (weights/threshold.json); AP numbers need no threshold.

usage: python evaluate_books.py
"""
import json

import numpy as np

from common import (OUT, HERE, load_pages, gts_of, evaluate, operating_point, bootstrap_ci, average_precision,
                    load_preds, COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR, ARTES_IMG_DIR, ARTES_ANN_DIR, EMPTY)

BOOKS = {"Compilatio (1489)": ("compilatio", COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR),
         "Artes Orandi (1485)": ("artes", ARTES_IMG_DIR, ARTES_ANN_DIR)}
MODELS = {"Detector v2 (+3 extra training books)": "mbv3_v2",
          "Detector v1 (Euclid only)": "mbv3_all",
          "Classical (proposals + GBM)": "classical_all"}
# the original dissertation model (best of its weight files on Euclid, chosen by compare.py)
_base = sorted(OUT.glob("preds_baseline_*_compilatio.json"))
if _base:
    MODELS["Original dissertation model (@ its own 0.99 cut-off)"] = _base[0].stem[len("preds_"):-len("_compilatio")]

cfg = HERE / "weights" / "threshold.json"
thr = json.loads(cfg.read_text())["threshold"] if cfg.exists() else 0.5
cthr_file = OUT / "classical_threshold.json"
cthr = json.loads(cthr_file.read_text())["threshold"] if cthr_file.exists() else 0.5
print(f"detector threshold {thr:.3f} (from Euclid CV), classical threshold {cthr:.3f} (from Euclid CV)")

rows, report = [], {}
for bname, (tag, img, ann) in BOOKS.items():
    gts = gts_of(load_pages(img, ann))
    n_caps = sum(len(v) for v in gts.values())
    for mname, mtag in MODELS.items():
        f = OUT / f"preds_{mtag}_{tag}.json"
        if not f.exists():
            print("missing", f.name); continue
        preds, meta = load_preds(f)
        t = thr if mtag.startswith("mbv3") else 0.99 if mtag.startswith("baseline") else cthr
        r = evaluate(gts, preds, t)
        (lo, hi), _ = bootstrap_ci(gts, [preds], lambda g, p: average_precision(g, p, 0.5), n=300)[0][0], None
        r.update(book=bname, model=mname, pages=len(gts), capitals=n_caps, AP50_CI=f"{lo:.3f}-{hi:.3f}",
                 sec_per_page=meta.get("sec_per_page"))
        rows.append(r)
        # error list for the write-up
        errs = []
        for pid, gt in gts.items():
            b, s = preds.get(pid, EMPTY); keep = s >= t
            single = operating_point({pid: gt}, {pid: (b[keep], s[keep])}, -np.inf)
            if single["FP"] or single["FN"]:
                errs.append((pid, single["FN"], single["FP"]))
        report[f"{bname} | {mname}"] = errs

cols = ["book", "model", "pages", "capitals", "AP50", "AP50_CI", "AP75", "AP50_95", "score_thr", "precision",
        "recall", "F1", "TP", "FP", "FN", "mean_IoU_matched", "pages_perfect", "sec_per_page"]
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |" for r in rows]
table = "\n".join(md)
print(table)
(OUT / "results_other_books.md").write_text(table + "\n")
(OUT / "errors_other_books.json").write_text(json.dumps(report, indent=1))
for k, v in report.items():
    print(k, "pages with errors (page, missed, false):", v)
