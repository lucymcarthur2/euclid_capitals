"""Score every available prediction file on the same Euclid pages and write the results table.

Operating point (precision / recall / F1): for each of the 5 page-block folds, the score cut-off is
the F1-optimal one on the OTHER four folds' pages, then applied to this fold - so no test page
influences its own threshold. The dissertation's fixed 0.99 cut-off is reported for the baseline too.

usage: python compare.py [--gt original|corrected]
"""
import argparse
import json

import numpy as np

from common import (OUT, HERE, load_pages, gts_of, kfold_ids, evaluate, average_precision,
                    operating_point, best_threshold, bootstrap_ci, load_preds, read_voc, EMPTY)

ap = argparse.ArgumentParser()
ap.add_argument("--gt", default="corrected", choices=["original", "corrected"])
args = ap.parse_args()

from common import ORIGINAL_ANN_DIR, EUCLID_IMG_DIR
pages = load_pages() if args.gt == "corrected" else load_pages(EUCLID_IMG_DIR, ORIGINAL_ANN_DIR)
gts = gts_of(pages)
folds = kfold_ids(pages)

MODELS = {
    "A  baseline (dissertation weights)": None,  # filled below with the best of the candidates
    "B  classical proposals + GBM": OUT / "preds_classical_cv.json",
    "C  MobileNetV3-FRCNN v2, real pages + 3 extra books (CV)": [OUT / f"preds_mbv3_v2_fold{k}.json" for k in range(5)],
    "C-v1  ... Euclid pages only (CV)": [OUT / f"preds_mbv3_fold{k}.json" for k in range(5)],
    "C+syn  ... + synthetic pages (fold ablation)": [OUT / f"preds_mbv3_syn_fold{k}.json" for k in range(5)],
}


def load_any(src):
    if isinstance(src, list):
        preds, meta = {}, {}
        for f in src:
            if f.exists():
                p, meta = load_preds(f); preds.update(p)
        return (preds, meta) if preds else (None, None)
    return load_preds(src) if src.exists() else (None, None)


def cv_operating_point(preds, ids_subset=None):
    kept = {}
    thrs = []
    for k, test in enumerate(folds):
        test = [i for i in test if ids_subset is None or i in ids_subset]
        if not test:
            continue
        others = {i: gts[i] for j, f in enumerate(folds) if j != k for i in f
                  if i in preds and (ids_subset is None or i in ids_subset)}
        thr = best_threshold(others, preds) if others else 0.5
        thrs.append(thr)
        for i in test:
            b, s = preds.get(i, EMPTY)
            kept[i] = (b[s >= thr], s[s >= thr])
    g = {i: gts[i] for i in kept}
    r = operating_point(g, kept, -np.inf)
    r["score_thr"] = f"per-fold {min(thrs):.2f}-{max(thrs):.2f}"
    return r


# baseline: score all four candidate weight files, keep the strongest (most generous to the original)
cands = {}
for tag in ["b4_models", "b4_root", "model_weights", "extract_capitals"]:
    f = OUT / f"preds_baseline_{tag}.json"
    if f.exists():
        p, m = load_preds(f)
        cands[tag] = (p, m, average_precision(gts, p, 0.5))
if cands:
    print("baseline candidates AP50:", {k: round(v[2], 4) for k, v in cands.items()})
    best_tag = max(cands, key=lambda k: cands[k][2])
    MODELS["A  baseline (dissertation weights)"] = OUT / f"preds_baseline_{best_tag}.json"
else:
    MODELS.pop("A  baseline (dissertation weights)")

rows, all_preds = [], {}
for name, src in MODELS.items():
    preds, meta = load_any(src)
    if preds is None:
        continue
    ids = set(preds)
    g = {i: gts[i] for i in gts if i in ids}
    res = evaluate(g, preds, 0.5)
    res.update(cv_operating_point(preds, ids))
    res.update(model=name, pages=len(g), capitals=int(sum(len(v) for v in g.values())),
               sec_per_page=meta.get("sec_per_page"), params=meta.get("params"))
    rows.append(res)
    all_preds[name] = preds
    if name.startswith("A"):
        d = operating_point(g, preds, 0.99)
        d.update(model="A  baseline @ dissertation cut-off 0.99", pages=len(g), capitals=res["capitals"],
                 AP50=res["AP50"], AP75=res["AP75"], AP50_95=res["AP50_95"],
                 sec_per_page=res["sec_per_page"], params=res["params"])
        rows.append(d)

# bootstrap 95% CIs (pages resampled): B (classical) vs C (detector), on pages both cover
names = [n for n in all_preds if n.startswith(("B", "C  "))]
cis = {}
if len(names) == 2:
    common_ids = set(all_preds[names[0]]) & set(all_preds[names[1]])
    g = {i: gts[i] for i in gts if i in common_ids}
    for label, fn in {"AP50": lambda gg, pp: average_precision(gg, pp, 0.5),
                      "AP75": lambda gg, pp: average_precision(gg, pp, 0.75)}.items():
        ci, diff = bootstrap_ci(g, [all_preds[n] for n in names], fn, n=500)
        cis[label] = {**{n.strip(): ci[k] for k, n in enumerate(names)}, "C minus B": diff}
    print("bootstrap 95% CI:", json.dumps(cis, indent=1))

# operating threshold for deployment = F1-optimal cut-off over ALL out-of-fold predictions of C
cname = next((n for n in all_preds if n.startswith("C  ")), None)
if cname and len(all_preds[cname]) == len(gts):
    thr = best_threshold(gts, all_preds[cname])
    (HERE / "weights").mkdir(exist_ok=True)
    (HERE / "weights" / "threshold.json").write_text(json.dumps({"threshold": thr, "chosen_by": "F1 on 5-fold out-of-fold predictions"}))
    print(f"deployment threshold {thr:.3f} -> weights/threshold.json")
bname = next((n for n in all_preds if n.startswith("B")), None)
if bname:
    (OUT / "classical_threshold.json").write_text(json.dumps({"threshold": best_threshold(gts, all_preds[bname])}))

cols = ["model", "pages", "capitals", "AP50", "AP75", "AP50_95", "score_thr", "precision", "recall", "F1",
        "TP", "FP", "FN", "mean_IoU_matched", "pages_perfect", "mean_abs_count_error", "sec_per_page", "params"]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.0f}"
    if isinstance(v, int) and v > 10000:
        return f"{v/1e6:.1f}M"
    return str(v)


md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in rows:
    md.append("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |")
table = "\n".join(md)
print(table)
(OUT / f"results_{args.gt}.md").write_text(table + "\n")
(OUT / f"results_{args.gt}.json").write_text(json.dumps({"rows": rows, "bootstrap": cis}, indent=1, default=str))

# precision-recall curves at IoU 0.5
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.5, 4.5))
for name, preds in all_preds.items():
    g = {i: gts[i] for i in gts if i in preds}
    from common import match_page
    sc, fl = [], []
    for pid, gt in g.items():
        b, s = preds[pid]; o = np.argsort(-s)
        tp, _ = match_page(gt, b[o], 0.5); sc += list(s[o]); fl += list(tp)
    o = np.argsort(-np.asarray(sc), kind="mergesort"); fl = np.asarray(fl)[o]
    n = sum(len(v) for v in g.values())
    ax.plot(np.cumsum(fl) / n, np.cumsum(fl) / np.arange(1, len(fl) + 1), label=name.strip())
ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0.5, 1.01); ax.set_ylim(0.5, 1.01)
ax.set_title(f"Euclid, IoU 0.5, {args.gt} labels"); ax.legend(fontsize=7, loc="lower left"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / f"pr_curves_{args.gt}.png", dpi=150)
print("wrote", OUT / f"results_{args.gt}.md")
