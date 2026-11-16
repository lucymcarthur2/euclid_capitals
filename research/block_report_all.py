"""Group ALL books' initials into physical blocks from blocks_all.py scores, for the Patterns page.

- Average-linkage clustering at SAME (0.70): a group's impressions must match each other ON AVERAGE,
  so one smudged look-alike cannot chain two blocks together.
- Mirror-image pairs never join (a block cannot print mirrored); strong mirror pairs are listed as
  possible recut copies.
- Each card is shown in the pose most of its impressions share; others are turned back to it.

usage: python block_report_all.py
Writes outputs/blocks_all.json (Patterns page), outputs/blocks_all_summary.csv, outputs/blocks_all_mirror_pairs.csv
"""
import csv
import json

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from common import CACHE, OUT
from block_matcher import ORIENT

SAME, UNCLEAR = 0.70, 0.50
z = np.load(CACHE / "blocks_all.npz")
books, pages, paths, boxes = z["books"], z["pages"], z["paths"], z["boxes"]
info = json.loads(str(z["book_info"]))
n = len(books)
pi, pj, po, ps = z["pi"], z["pj"], z["po"], z["ps"]
O = z["cheap_orient"]  # O[j, i]: turn of j that best fits i (cheap check)

Sd = np.full((n, n), -1.0, np.float32)
Sd[pi, pj] = Sd[pj, pi] = ps
mirror = po >= 4
Sc = Sd.copy()
Sc[pi[mirror], pj[mirror]] = Sc[pj[mirror], pi[mirror]] = -1  # mirrored match != same block
D = 1 - Sc
np.fill_diagonal(D, 0)
lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=1 - SAME, criterion="distance")

ratdolt = {b for b, (p, _) in info.items() if p == "Ratdolt"}
groups = {}
for i, c in enumerate(lab):
    groups.setdefault(int(c), []).append(i)

out_groups, meta = [], {}
for gid, (c, mem) in enumerate(sorted(groups.items(), key=lambda x: -len(x[1]))):
    mem = np.array(mem)
    # reference pose: the member that most others fit without turning (majority = printed upright)
    cand = mem[:60]
    ref = max(cand, key=lambda r: ((O[mem, r] == 0).sum(), Sc[mem, r].sum()))
    members = []
    for m in mem:
        others = mem[mem != m]
        known = Sc[m, others][Sc[m, others] > -1]
        s = float(known.mean()) if len(known) else None
        o = 0 if m == ref else int(O[m, ref])
        if len(mem) == 1:  # a lone initial: is it close to any existing block?
            row = Sc[m].copy(); row[m] = -1
            j = int(row.argmax())
            s, verdict, best = float(row[j]), ("unclear" if row[j] >= UNCLEAR else "no match"), j
        else:
            verdict, best = ("same block" if s is None or s >= SAME else "unclear"), None
        members.append(dict(item=int(m), orient=o, score=None if s is None else round(s, 3), verdict=verdict,
                            best_match=best))
    out_groups.append(dict(id=gid, members=members))

items = [dict(id=f"{books[i]}/{pages[i]}#{i}", book=str(books[i]), page=str(pages[i]),
              box=[int(v) for v in boxes[i]], path=str(paths[i])) for i in range(n)]
data = dict(name="all", books=list(dict.fromkeys(str(b) for b in books)),
            book_info={b: dict(printer=p, date=d) for b, (p, d) in info.items()},
            ratdolt_books=sorted(ratdolt), thresholds=dict(same=SAME, unclear=UNCLEAR), orientations=ORIENT,
            items=items, groups=out_groups)
(OUT / "blocks_all.json").write_text(json.dumps(data))

# ---- summary: per book, how many initials sit in a block that a Ratdolt book also used
gbooks = [{str(books[m["item"]]) for m in g["members"]} for g in out_groups]
item_group = {m["item"]: k for k, g in enumerate(out_groups) for m in g["members"]}
rows = []
for b in data["books"]:
    idx = [i for i in range(n) if books[i] == b]
    with_r = sum(1 for i in idx if (gbooks[item_group[i]] - {b}) & ratdolt)
    shared_blocks = {item_group[i] for i in idx if (gbooks[item_group[i]] - {b}) & ratdolt}
    rows.append(dict(book=b, printer=info[b][0], initials=len(idx), blocks=len({item_group[i] for i in idx}),
                     initials_from_a_block_ratdolt_also_used=with_r, such_blocks=len(shared_blocks)))
with open(OUT / "blocks_all_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{n} initials -> {len(out_groups)} blocks; {sum(len(g) > 1 for g in gbooks)} used in more than one book; "
      f"{sum(bool(g & ratdolt) and bool(g - ratdolt) for g in gbooks)} shared between Ratdolt and another printer")
for r in rows:
    print(f"  {r['book']:<26} {r['printer']:<24} {r['initials']:>4} initials, {r['blocks']:>3} blocks; "
          f"{r['initials_from_a_block_ratdolt_also_used']:>4} from a block Ratdolt also used ({r['such_blocks']} blocks)")

# ---- strong mirror-image matches: candidate recut (reversed) copies
with open(OUT / "blocks_all_mirror_pairs.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["a", "b", "score", "orientation"])
    for k in np.where(mirror & (ps >= SAME))[0]:
        w.writerow([items[pi[k]]["id"], items[pj[k]]["id"], f"{ps[k]:.3f}", ORIENT[po[k]]])
print("mirror-image matches >= SAME:", int((mirror & (ps >= SAME)).sum()))
