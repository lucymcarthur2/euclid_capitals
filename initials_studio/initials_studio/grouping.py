"""Group capitals into woodblock patterns, keep the user's edits, and derive per-capital flags.

groups.json = {"groups": [{"id": int, "members": [{"key": "book/cap", "orient": int, "score": float|None,
                "verdict": "same"|"unclear"|"alone"}]}], "next_id": int, "params": {...}, "updated": t}
"""
import time

import numpy as np

from .config import GROUPING_DIR, load_settings
from . import store
from .extract import visible
from .matcher import book_features, score_pairs

GROUPS_FILE = GROUPING_DIR / "groups.json"


def current_keys(s, books=None):
    """every visible capital in the given books (default: every finished book), sorted."""
    keys = []
    for b in store.list_books():
        if b.get("status") not in ("detected", "example") or (books is not None and b["id"] not in books):
            continue
        for c in store.get_capitals(b["id"]):
            if visible(c, s):
                keys.append(f"{b['id']}/{c['id']}")
    return sorted(keys)


def match_status(s):
    """per book: capitals and how many are not in any woodblock group yet; plus which pairs of books
    have already been compared with each other (so the page knows when a selection needs work)."""
    g = load_groups()
    grouped = {m["key"] for grp in g["groups"] for m in grp["members"]}
    books = {}
    for b in store.list_books():
        if b.get("status") not in ("detected", "example"):
            continue
        keys = current_keys(s, {b["id"]})
        n_in = sum(k in grouped for k in keys)
        books[b["id"]] = dict(capitals=len(keys), ungrouped=len(keys) - n_in)
    return dict(books=books, compared_pairs=g.get("compared_pairs", []))


def load_groups():
    return store.read_json(GROUPS_FILE, {"groups": [], "next_id": 0})


def save_groups(g):
    g["updated"] = time.time()
    store.write_json(GROUPS_FILE, g)


def _features(keys):
    by_book = {}
    for k in keys:
        b, c = k.split("/")
        by_book.setdefault(b, []).append(c)
    feats = {}
    for b, cs in by_book.items():
        for c, f in zip(cs, book_features(b, cs)):
            feats[f"{b}/{c}"] = f
    G = np.stack([feats[k][0] for k in keys]) if keys else np.zeros((0, 160, 160), np.uint8)
    INK = np.stack([feats[k][1] for k in keys]) if keys else np.zeros((0, 160, 160), np.uint8)
    return G, INK


def _dense(keys, pairs):
    n = len(keys)
    idx = {k: i for i, k in enumerate(keys)}
    Sm = np.full((n, n), -1.0, np.float32)
    mirror = np.zeros((n, n), bool)
    turn = np.zeros((n, n), np.int8)  # turn[j, i]: orientation of j that fits i
    for (a, b), (s, o) in pairs.items():
        i, j = idx.get(a), idx.get(b)
        if i is None or j is None:
            continue
        Sm[i, j] = Sm[j, i] = s
        turn[j, i] = o
        turn[i, j] = o if o >= 4 else (4 - o) % 4  # inverse turn
        if o >= 4:
            mirror[i, j] = mirror[j, i] = True
    return Sm, mirror, turn


def _ref_and_orients(members, turn, O):
    """choose the reference impression most others fit without turning; orientation of each vs it."""
    mem = np.array(members)
    def known_turn(m, r):
        return int(turn[m, r]) if turn[m, r] or turn[r, m] else int(O[m, r])
    ref = max(mem[:60], key=lambda r: sum(known_turn(m, r) == 0 for m in mem))
    return {int(m): (0 if m == ref else known_turn(m, ref)) for m in mem}


def regroup(progress=lambda f, m: None, cancel=lambda: False, keep_edits=True, books=None):
    """Compare the capitals of `books` (default: all) - scoring only pairs never scored before - then
    keep existing woodblock groups (with the user's edits) and place not-yet-grouped capitals into
    them when they match well enough; capitals that fit nowhere are clustered among themselves.
    Groups' members from books outside `books` are left exactly as they are.
    keep_edits=False re-clusters the chosen books' capitals from scratch."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    s = load_settings()
    keys = current_keys(s, set(books) if books else None)
    every = set(current_keys(s))  # all capitals that still exist (to drop deleted / thrown-out ones)
    if len(keys) < 2:
        old = load_groups()
        have = {m["key"] for g in old["groups"] for m in g["members"]}
        nid = old.get("next_id", 0)
        for k in keys:
            if k not in have:
                old["groups"].append(dict(id=nid, members=[dict(key=k, orient=0, score=None, verdict="alone")])); nid += 1
        old["next_id"] = nid
        save_groups(old)
        return
    progress(0.01, f"Preparing {len(keys):,} capitals…")
    G, INK = _features(keys)
    done_books = frozenset(load_groups().get("compared_pairs", []))
    pairs, O = score_pairs(keys, G, INK, s, progress, cancel, done_books)
    if cancel():
        return
    progress(0.96, "Grouping into woodblock patterns…")
    Sm, mirror, turn = _dense(keys, pairs)
    Sc = np.where(mirror, -1, Sm)
    idx = {k: i for i, k in enumerate(keys)}
    same = float(s["same_block"])

    old = load_groups()
    prev = {m["key"]: m for g in old["groups"] for m in g["members"]}  # stored member info
    assigned, groups = {}, []
    for g in old["groups"]:  # existing groups, minus capitals that no longer exist
        mem = [m["key"] for m in g["members"] if m["key"] in every]
        if not keep_edits:  # from scratch: the chosen books' capitals leave their groups
            mem = [k for k in mem if k not in idx]
        if mem:
            groups.append(dict(id=g["id"], keys=mem, manual=g.get("manual", False)))
            for k in mem:
                assigned[k] = g["id"]
    next_id = old.get("next_id", 0)
    new = [k for k in keys if k not in assigned]
    # place new capitals into existing groups by average score
    leftover = []
    if groups and new:
        gi = [np.array([idx[k] for k in g["keys"] if k in idx], dtype=int) for g in groups]
        for k in new:
            i = idx[k]
            best, bs = None, -1
            for n, mem in enumerate(gi):
                if not len(mem):
                    continue
                v = Sc[i, mem]; v = v[v > -1]
                if len(v) and v.mean() > bs:
                    best, bs = n, float(v.mean())
            if best is not None and bs >= same:
                groups[best]["keys"].append(k)
            else:
                leftover.append(k)
    else:
        leftover = new
    # cluster the rest among themselves (average linkage: no chaining of look-alikes)
    if len(leftover) == 1:
        groups.append(dict(id=next_id, keys=leftover)); next_id += 1
    elif leftover:
        li = np.array([idx[k] for k in leftover])
        D = 1 - Sc[np.ix_(li, li)]
        np.fill_diagonal(D, 0)
        D = (D + D.T) / 2
        lab = fcluster(linkage(squareform(D, checks=False), method="average"), t=1 - same, criterion="distance")
        for c in sorted(set(lab)):
            groups.append(dict(id=next_id, keys=[leftover[n] for n in np.where(lab == c)[0]])); next_id += 1

    out = []
    for g in groups:
        mem = [idx[k] for k in g["keys"] if k in idx]
        if not mem:  # a group with no capitals from the chosen books: keep it exactly as stored
            out.append(dict(id=g["id"], members=[prev[k] for k in g["keys"]], manual=g.get("manual", False)))
            continue
        ori = _ref_and_orients(mem, turn, O)
        members = []
        for k in g["keys"]:
            if k not in idx:  # member from a book not in this run: keep its stored values
                members.append(prev[k])
                continue
            i = idx[k]
            others = [m for m in mem if m != i]
            v = Sc[i, others]; v = v[v > -1]
            sc = float(v.mean()) if len(v) else None
            if len(mem) == 1:
                row = Sc[i].copy(); row[i] = -1
                sc = float(row.max())
                verdict = "unclear" if sc >= s["unclear_band"] else "alone"
            else:
                verdict = "same" if sc is None or sc >= same else "unclear"
            members.append(dict(key=k, orient=ori[i], score=None if sc is None else round(sc, 3), verdict=verdict))
        out.append(dict(id=g["id"], members=members, manual=g.get("manual", False)))
    out.sort(key=lambda g: -len(g["members"]))
    # strong mirror-image matches = candidate recut copies (kept from earlier runs for other books)
    mirrors = [p for p in old.get("mirror_pairs", []) if not (p["a"] in idx and p["b"] in idx)
               and p["a"] in every and p["b"] in every]
    mirrors += [dict(a=keys[i], b=keys[j], score=round(float(Sm[i, j]), 3))
                for i, j in zip(*np.where(np.triu(mirror & (Sm >= same), 1)))]
    chosen = sorted({k.split("/")[0] for k in keys})
    done = set(old.get("compared_pairs", [])) | {f"{a}|{b}" for a in chosen for b in chosen if a <= b}
    save_groups({"groups": out, "next_id": next_id, "params": s, "mirror_pairs": mirrors,
                 "compared_pairs": sorted(done)})
    progress(1.0, f"Compared {len(keys):,} capitals from {len(chosen)} books; {len(out):,} woodblocks in total")


def group_flags():
    """capital key -> extra flags from grouping (shown alongside detection flags)."""
    g = load_groups()
    out = {}
    for grp in g["groups"]:
        for m in grp["members"]:
            if m["verdict"] == "unclear":  # (a capital matching nothing else is normal, so not flagged)
                out.setdefault(m["key"], []).append("unclear_block")
    for p in g.get("mirror_pairs", []):
        for k in (p["a"], p["b"]):
            out.setdefault(k, []).append("mirror")
    return out


def apply_edit(op):
    """User edits from the Patterns page: move capitals to a group / to a new group."""
    g = load_groups()
    keys = set(op.get("keys", []))
    if not keys:
        return g
    moved = []
    for grp in g["groups"]:
        keep = []
        for m in grp["members"]:
            (moved if m["key"] in keys else keep).append(m)
        grp["members"] = keep
    for m in moved:
        m["verdict"] = "same"; m["score"] = m.get("score")
    if op["op"] == "move":
        tgt = next(x for x in g["groups"] if x["id"] == op["target"])
        tgt["members"] += moved; tgt["manual"] = True
    elif op["op"] == "new":
        g["groups"].append(dict(id=g["next_id"], members=moved, manual=True)); g["next_id"] += 1
    g["groups"] = [x for x in g["groups"] if x["members"]]
    g["groups"].sort(key=lambda x: -len(x["members"]))
    save_groups(g)
    return g
