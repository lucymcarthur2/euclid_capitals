"""Same-block matching and grouping.

Two impressions of one physical woodblock share every detail of the cut design. Each capital is
cropped to its black frame, squared to 160x160, and its ink pattern compared with another's after
fine alignment (OpenCV ECC). Score ~0.9 = same block, < 0.5 = different design (calibrated on
Euclid, where Ratdolt reused each block many times). Blocks can be printed turned 90/180/270 degrees;
a MIRROR-image match can never be the same block, so it is reported as a possible recut copy.

Speed: a cheap 8-orientation comparison of tiny ink maps picks each capital's closest candidates;
only those are fully aligned. Pair scores are cached, so adding a book only scores new pairs.
"""
import time

import cv2
import numpy as np
from PIL import Image

from .config import GROUPING_DIR
from . import store

S, B = 160, 8
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4)
CHEAP_T = 0.45
ORIENT = ["upright", "turned 90°", "turned 180°", "turned 270°", "mirrored", "mirrored + 90°",
          "mirrored + 180°", "mirrored + 270°"]


def orient(x, k):
    if k >= 4:
        x = x[:, ::-1]
    return np.ascontiguousarray(np.rot90(x, -(k % 4)))


def tight(grey):
    ink = cv2.threshold(cv2.GaussianBlur(grey, (5, 5), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n < 2:
        return grey
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = st[k, :4]
    if w * h < 0.3 * grey.size:
        return grey
    return grey[y:y + h, x:x + w]


def features(crop_file):
    """-> (g uint8 160x160 contrast-normalised grey, ink uint8 0/255) for one capital crop."""
    grey = np.asarray(Image.open(crop_file).convert("L"))
    g = cv2.resize(tight(grey), (S, S), interpolation=cv2.INTER_AREA)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)
    ink = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return g, ink


def _maps(g, ink):
    return (cv2.GaussianBlur(g.astype(np.float32), (0, 0), 2),
            cv2.GaussianBlur(ink.astype(np.float32) / 255, (0, 0), 1.2))


def cheap_vectors(ink):
    m = cv2.GaussianBlur(ink.astype(np.float32) / 255, (0, 0), 1.2)
    x0 = cv2.GaussianBlur(cv2.resize(m, (24, 24), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    out = []
    for k in range(8):
        x = orient(x0, k)[2:-2, 2:-2]
        x = x - x.mean()
        out.append((x / (np.linalg.norm(x) + 1e-9)).ravel())
    return np.stack(out).astype(np.float32)


def ncc(x, y):
    x = x - x.mean(); y = y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def full_score(a, b, o):
    """a, b: (gf, m) maps. Score of a turned by o, aligned onto b."""
    ga, ma = orient(a[0], o), orient(a[1], o)
    M = np.eye(2, 3, dtype=np.float32)
    try:
        _, M = cv2.findTransformECC(b[0], ga, M, cv2.MOTION_AFFINE, CRIT, None, 5)
    except cv2.error:
        M = np.eye(2, 3, dtype=np.float32)
    wa = cv2.warpAffine(ma, M, (S, S), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return ncc(wa[B:-B, B:-B], b[1][B:-B, B:-B])


# ---------------- worker side (multiprocessing, spawn-safe) ----------------
_W = None


def _init(G, INK):
    global _W
    _W = (G, INK)
    cv2.setNumThreads(1)


def _pair(job):
    i, j, o, o2 = job
    G, INK = _W
    a, b = _maps(G[j], INK[j]), _maps(G[i], INK[i])
    s = full_score(a, b, o)
    if o >= 4:  # cheap check preferred a mirror: near-symmetric designs - try the best plain turn too
        s2 = full_score(a, b, o2)
        if s2 >= s - 0.05:
            return i, j, o2, s2
    return i, j, o, s


# ---------------- feature + pair caches ----------------
def _feat_file(book_id):
    return GROUPING_DIR / "features" / f"{book_id}.npz"


def book_features(book_id, cap_ids):
    """cached (g, ink) per capital id; computes missing ones from the crops."""
    f = _feat_file(book_id)
    cache = {}
    if f.exists():
        z = np.load(f, allow_pickle=False)
        ids, g, ink = z["ids"], z["g"], z["ink"]  # read each array once (indexing z[...] re-decompresses)
        cache = {k: (g[n], ink[n]) for n, k in enumerate(ids)}
    missing = [c for c in cap_ids if c not in cache]
    for c in missing:
        cache[c] = features(store.crop_path(book_id, c))
    if missing:
        ids = sorted(cache)
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, ids=np.array(ids), g=np.stack([cache[k][0] for k in ids]),
                            ink=np.stack([cache[k][1] for k in ids]))
    return [cache[c] for c in cap_ids]


def _pairs_file(rot):
    return GROUPING_DIR / f"pairs_{'rot' if rot else 'norot'}.npz"


def load_pairs(rot):
    f = _pairs_file(rot)
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=False)
    return {(a, b): (float(s), int(o)) for a, b, s, o in zip(z["a"], z["b"], z["s"], z["o"])}


def save_pairs(rot, pairs):
    f = _pairs_file(rot); f.parent.mkdir(parents=True, exist_ok=True)
    keys = list(pairs)
    np.savez_compressed(f, a=np.array([k[0] for k in keys]), b=np.array([k[1] for k in keys]),
                        s=np.array([pairs[k][0] for k in keys], np.float32),
                        o=np.array([pairs[k][1] for k in keys], np.int8))


def score_pairs(keys, G, INK, s, progress=lambda f, m: None, cancel=lambda: False, done_books=frozenset()):
    """Make sure every candidate pair among `keys` (global capital keys 'book/cap') has a full score.
    Pairs between two books listed together in done_books ("a|b", sorted) were already compared in an
    earlier run and are skipped, so adding a book only compares that book.
    Returns (pairs dict, cheap orientation matrix O[j, i])."""
    rot = bool(s["allow_rotations"])
    n = len(keys)
    progress(0.02, "Comparing capitals (quick check)…")
    V = np.stack([cheap_vectors(x) for x in INK]) if n else np.zeros((0, 8, 400), np.float32)
    allowed = list(range(8)) if rot else [0, 4]
    U = V[:, 0, :]
    C = np.full((n, n), -9, np.float32); O = np.zeros((n, n), np.int8); O2 = np.zeros((n, n), np.int8)
    for st in range(0, n, 256):
        P = np.einsum("nkd,md->nmk", V[st:st + 256][:, allowed], U)
        C[st:st + 256] = P.max(2)
        O[st:st + 256] = np.array(allowed)[P.argmax(2)]
        plain = [k for k, a in enumerate(allowed) if a < 4]
        O2[st:st + 256] = np.array(allowed)[plain][P[:, :, plain].argmax(2)]
    Cs = np.maximum(C, C.T); np.fill_diagonal(Cs, -9)
    K = min(int(s["neighbours"]), max(n - 1, 1))
    keep = np.zeros((n, n), bool)
    if n > 1:
        top = np.argpartition(-Cs, K - 1, axis=1)[:, :K]
        keep[np.arange(n)[:, None], top] = True
    keep |= Cs >= CHEAP_T; keep |= keep.T
    np.fill_diagonal(keep, False)
    pairs = load_pairs(rot)
    I, J = np.where(np.triu(keep, 1))
    bk = [k.split("/")[0] for k in keys]
    def done(i, j):
        a, b = sorted((bk[i], bk[j]))
        return f"{a}|{b}" in done_books
    jobs = [(i, j, int(O[j, i]), int(O2[j, i])) for i, j in zip(I, J)
            if (keys[i], keys[j]) not in pairs and not done(i, j)]
    if jobs:
        from multiprocessing import get_context
        t0 = time.time()
        with get_context("spawn").Pool(int(s["workers"]), initializer=_init, initargs=(G, INK)) as pool:
            for k, (i, j, o, sc) in enumerate(pool.imap_unordered(_pair, jobs, chunksize=32)):
                pairs[(keys[i], keys[j])] = (sc, o)
                if k % 500 == 0:
                    el = time.time() - t0
                    eta = el / max(k, 1) * (len(jobs) - k)
                    progress(0.05 + 0.9 * k / len(jobs),
                             f"Matching woodblocks: {k:,} of {len(jobs):,} comparisons (about {eta / 60:.0f} min left)")
                    if cancel():
                        pool.terminate(); break
                    if k and k % 20000 == 0:
                        save_pairs(rot, pairs)
        save_pairs(rot, pairs)
    return pairs, O
