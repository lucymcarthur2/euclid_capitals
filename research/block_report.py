"""Report from block_matcher.py scores: which initials of another book were printed from the same
woodblocks as Ratdolt's Euclid.

Thresholds (fixed from the Euclid-only calibration, where pair scores are bimodal):
  >= SAME (0.70): same physical block      0.50-0.70: unclear (worn, damaged or a close recut copy)
  <  0.50: different design

Euclid impressions are first grouped into blocks: crops joined by a same-block score are one block
(connected components). Each other-book initial is then assigned to the block of its best match.

usage: python block_report.py NAME
Outputs: outputs/blocks_<NAME>.csv, outputs/blocks_<NAME>_matches_*.jpg, outputs/blocks_<NAME>_unmatched.jpg
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import CACHE, OUT
from block_matcher import crop, prep, S, CRIT, ORIENT, orient

SAME, UNCLEAR = 0.70, 0.50
name = sys.argv[1] if len(sys.argv) > 1 else "hyginus"
z = np.load(CACHE / f"blocks_{name}.npz")
score, nE, turn = z["score"], int(z["n_euclid"]), z["turn"]
ids, paths, boxes = z["ids"], z["paths"], z["boxes"]
nO = len(ids) - nE

# --- Euclid blocks: connected components of the same-block graph
EE = np.nan_to_num(score[:nE, :nE], nan=-1)
TE = turn[:nE, :nE]
not_mirror = (TE < 4) & (TE.T < 4)  # mirror-image pairs cannot be one physical block
# AVERAGE linkage, not chaining: two groups merge only if their impressions match on average
# (>= SAME). Chaining (connected components) glued different S blocks together through a single
# in-between impression.
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
Dm = 1 - np.where(not_mirror, np.clip(EE, -1, 1), -1)
Dm = (Dm + Dm.T) / 2
np.fill_diagonal(Dm, 0)
lab = fcluster(linkage(squareform(Dm, checks=False), method="average"), t=1 - SAME, criterion="distance")
order_ = {c: k for k, c in enumerate(sorted(set(lab), key=lambda c: -(lab == c).sum()))}
block = np.array([order_[c] for c in lab])
sizes = np.bincount(block)
nn_E = EE.max(1)
print(f"Euclid: {nE} impressions -> {len(sizes)} distinct blocks "
      f"({(sizes > 1).sum()} used more than once; {(sizes == 1).sum()} seen once)")
print("Euclid nearest-neighbour score: median %.2f; share >= %.2f: %.0f%%" % (np.median(nn_E), SAME, 100 * (nn_E >= SAME).mean()))

# --- other book vs Euclid
OE = np.nan_to_num(score[nE:, :nE], nan=-1)
# score against a whole Euclid block = average over its impressions (not the single best one),
# so one look-alike impression cannot pull an initial into the wrong block
blk_score = np.stack([OE[:, block == b].mean(1) for b in range(len(sizes))], 1)
best_b = blk_score.argmax(1); best = blk_score.max(1)
best_j = np.array([np.flatnonzero(block == best_b[k])[OE[k, block == best_b[k]].argmax()] for k in range(nO)])
ori = np.array([int(turn[nE + k, best_j[k]]) for k in range(nO)])  # how the other-book initial is turned
cls = np.where(best >= SAME, "same block", np.where(best >= UNCLEAR, "unclear", "different design"))
# a mirror image cannot come from the same physical block: it is a recut (reversed) copy
cls = np.where((ori >= 4) & (best >= UNCLEAR), "mirror image: recut copy", cls)
print("orientations of same-block matches:", {ORIENT[o]: int(((cls == "same block") & (ori == o)).sum())
                                             for o in range(4) if ((cls == "same block") & (ori == o)).any()})
for c in ("same block", "unclear", "mirror image: recut copy", "different design"):
    print(f"{name}: {c}: {(cls == c).sum()} of {nO}")
matched_blocks = sorted({int(block[best_j[k]]) for k in range(nO) if cls[k] == "same block"})
print(f"{name} uses {len(matched_blocks)} of Euclid's {len(sizes)} blocks")

with open(OUT / f"blocks_{name}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["initial", "best_euclid_match", "score", "verdict", "orientation", "euclid_block",
                "euclid_block_impressions"])
    for k in np.argsort(-best):
        w.writerow([ids[nE + k], ids[best_j[k]], f"{best[k]:.3f}", cls[k], ORIENT[ori[k]],
                    int(block[best_j[k]]), int(sizes[block[best_j[k]]])])

# --- pictures: other-book initial | Euclid match | overlay after alignment
font = ImageFont.load_default(size=15)
T = 150


def tile(k):
    a = prep(crop(paths[nE + k], boxes[nE + k])); b = prep(crop(paths[best_j[k]], boxes[best_j[k]]))
    a = {key: orient(v, ori[k]) for key, v in a.items()}  # show it turned the way it matched
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(b["gf"], a["gf"], M, cv2.MOTION_AFFINE, CRIT, None, 5)
    except cv2.error:
        pass
    wa = cv2.warpAffine(a["g"], M, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # each print's own ink map (Otsu per crop), so heavier inking in one book doesn't paint it all red
    ia = cv2.warpAffine(a["m"], M, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP) > 0.5
    ib = b["m"] > 0.5
    a = dict(a, g=wa)  # show the other-book initial aligned too
    ov = np.full((S, S, 3), 255, np.uint8)
    ov[ia & ib] = (40, 40, 40)       # ink in both: dark grey
    ov[ia & ~ib] = (220, 30, 30)     # only in the other book: red
    ov[~ia & ib] = (30, 140, 220)    # only in Euclid: blue
    row = Image.new("RGB", (3 * T + 20, T + 22), "white")
    for c, im in enumerate([a["g"], b["g"], ov]):
        row.paste(Image.fromarray(im).resize((T, T)), (c * (T + 10), 22))
    turned = "" if ori[k] == 0 else f" [{ORIENT[ori[k]]}]"
    lab = f"{ids[nE + k].split('/')[1]}  vs  {ids[best_j[k]].split('/')[1]}   {best[k]:.2f}  {cls[k]}{turned}"
    ImageDraw.Draw(row).text((2, 2), lab, fill="black", font=font)
    return row


def sheet(ks, fname, title):
    if not len(ks):
        return
    cols, rows_per = 2, 8
    for s in range(0, len(ks), cols * rows_per):
        part = ks[s:s + cols * rows_per]
        tw, th = 3 * T + 20, T + 22
        sh = Image.new("RGB", (cols * (tw + 20), 30 + ((len(part) + cols - 1) // cols) * (th + 8)), "white")
        ImageDraw.Draw(sh).text((4, 6), title + "   [left: %s | middle: best Euclid match | right: overlay, grey = both, red = %s only, blue = Euclid only]" % (name, name), fill="black", font=font)
        for i, k in enumerate(part):
            sh.paste(tile(k), ((i % cols) * (tw + 20), 30 + (i // cols) * (th + 8)))
        sh.save(OUT / f"{fname}_{s // (cols * rows_per)}.jpg", quality=88)


# --- groups for the review tool's Patterns view: every crop (Euclid + other book) in one list;
# a group = one physical block. Other-book initials join the Euclid block they match (same block,
# not mirrored); the rest start as their own group, so the reviewer can merge or split by eye.
import json
items, groups = [], {}
for i in range(len(ids)):
    book, rest = str(ids[i]).split("/", 1)
    items.append(dict(id=str(ids[i]), book=book, page=rest.split("#")[0], box=[int(v) for v in boxes[i]],
                      path=str(paths[i])))
for i in range(nE):
    groups.setdefault(int(block[i]), []).append(dict(item=i, orient=0, score=None))
nxt = len(sizes)
for k in range(nO):
    if cls[k] == "same block":
        g = int(block[best_j[k]])
    else:
        g, nxt = nxt, nxt + 1
    groups.setdefault(g, []).append(dict(item=nE + k, orient=int(ori[k]), score=round(float(best[k]), 3),
                                         best_match=int(best_j[k]), verdict=str(cls[k])))
data = dict(name=name, books=sorted({it["book"] for it in items}), thresholds=dict(same=SAME, unclear=UNCLEAR),
            orientations=ORIENT, items=items,
            groups=[dict(id=g, members=m) for g, m in sorted(groups.items(), key=lambda x: -len(x[1]))])
(OUT / f"blocks_{name}.json").write_text(json.dumps(data))
print("wrote", OUT / f"blocks_{name}.json", f"({len(data['groups'])} groups)")

order = np.argsort(-best)
sheet([k for k in order if cls[k] == "same block"], f"blocks_{name}_same", "SAME BLOCK")
sheet([k for k in order if cls[k] != "same block"], f"blocks_{name}_notsame", "UNCLEAR / DIFFERENT")
print("wrote", OUT / f"blocks_{name}.csv", "and sheets outputs/blocks_%s_*.jpg" % name)
