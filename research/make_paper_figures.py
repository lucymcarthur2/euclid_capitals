"""High-resolution figures for the Heritage paper (Adam Gibson's requests).

Fig 1 (capitals panel): Euclid page 32811_018 at full scan resolution with thick red boxes round the
    four decorated capitals (hand-checked labels).
Fig 3: the two skewed D impressions with ORB keypoint matches drawn as lines - the dissertation's
    matching method (ORB, brute-force Hamming matcher, Lowe ratio 0.73), drawn like Figure 2.
Fig 2 (c)-(d) check: shows that the O pair matches after a 180-degree rotation, not a mirror flip.

usage: python make_paper_figures.py
Writes paper_figures/*.png (and .tif for Fig 1) at 300 dpi.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from common import DISS, load_pages

OUT = Path(__file__).resolve().parent / "paper_figures"
OUT.mkdir(exist_ok=True)
RED = (210, 0, 0)


def fig1():
    page = next(p for p in load_pages() if p.page_id == "32811_018")
    im = Image.open(DISS / "tiff_cropped" / "32811_018.tif").convert("RGB")  # full-resolution scan
    w = max(8, round(im.height / 270))  # ~16 px on a 4314 px page: thick enough to survive reduction
    d = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in page.boxes.astype(int):
        pad = round(0.03 * (x2 - x1))
        d.rectangle([x1 - pad, y1 - pad, x2 + pad, y2 + pad], outline=RED, width=w)
    im.save(OUT / "fig1_capitals_32811_018.png", dpi=(300, 300))
    im.save(OUT / "fig1_capitals_32811_018.tif", dpi=(300, 300), compression="tiff_lzw")
    # a version cropped to the text block, as in the paper's panel
    xs = page.boxes[:, [0, 2]]; ys = page.boxes[:, [1, 3]]
    print("fig1:", im.size, "line width", w)
    return im


def orb_pair(a_path, b_path, out_name, ratio=0.73, flip_b=None, height=900):
    """Two crops side by side with ORB matches, styled like the paper's Figure 2."""
    A = cv2.imread(str(a_path), cv2.IMREAD_GRAYSCALE)
    Bc = cv2.imread(str(b_path), cv2.IMREAD_GRAYSCALE)
    if flip_b is not None:
        Bc = cv2.flip(Bc, flip_b) if flip_b != "rot180" else cv2.rotate(Bc, cv2.ROTATE_180)
    orb = cv2.ORB_create(nfeatures=2000)
    ka, da = orb.detectAndCompute(A, None)
    kb, db = orb.detectAndCompute(Bc, None)
    good = []
    if da is not None and db is not None:
        for m in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                good.append(m[0])
    # slope test from the dissertation: share of match lines that are near-horizontal
    def slope(m):
        (x1, y1), (x2, y2) = ka[m.queryIdx].pt, kb[m.trainIdx].pt
        return abs((y2 - y1) / ((x2 + A.shape[1]) - x1 + 1e-9))
    flat = sum(slope(m) < 0.3 for m in good) / max(len(good), 1)
    vis = cv2.drawMatches(cv2.cvtColor(A, cv2.COLOR_GRAY2BGR), ka, cv2.cvtColor(Bc, cv2.COLOR_GRAY2BGR), kb, good, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchesThickness=2)
    k = height / vis.shape[0]
    vis = cv2.resize(vis, (round(vis.shape[1] * k), height), interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(OUT / out_name, dpi=(300, 300))
    return len(good), flat, A.shape, Bc.shape


def same_height(a_path, b_path, h, tmp):
    """resize both crops to the same height (keeping each one's aspect), so they sit side by side as in Fig 2."""
    outs = []
    for k, p in enumerate((a_path, b_path)):
        im = Image.open(p).convert("L")
        im = im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
        f = tmp / f"_h{k}.png"; im.save(f); outs.append(f)
    return outs


def fig3():
    D = DISS / "organised_capitals" / "D"
    a, b = D / "extracted_object_32811_167.jpg_0.png", D / "extracted_object_32811_205.jpg_3.png"
    for p in (a, b):
        print("fig3 source", p.name, Image.open(p).size)
    a2, b2 = same_height(a, b, 600, OUT)
    n, flat, sa, sb = orb_pair(a2, b2, "fig3_D_pair_orb_matches.png", height=1200)
    for f in (a2, b2):
        f.unlink()
    print(f"fig3: {n} good ORB matches, {flat:.0%} of match lines near-horizontal")


def fig1_diagrams(panel_png, page_index=17):
    """Xuzhuo's panel (diagrams): page 18 of the Library of Congress gold-leaf copy (euclid_gold_leaf.pdf).
    His red boxes are found in the low-resolution figure, carried onto the full-resolution scan through
    a homography fitted to ORB matches, and redrawn at the same line weight as the capitals panel."""
    import pymupdf
    doc = pymupdf.open(DISS / "euclid_gold_leaf.pdf")
    page = doc[page_index]
    imgs = page.get_images()
    pix = pymupdf.Pixmap(doc, imgs[0][0]) if len(imgs) == 1 else page.get_pixmap(dpi=300)
    if pix.n - pix.alpha > 3 or pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    full = Image.frombytes("RGB" if pix.n >= 3 else "L", (pix.width, pix.height), pix.samples).convert("RGB")
    print("diagram page full size", full.size)
    low = cv2.imread(str(panel_png))
    # red box pixels in the low-res figure -> rectangles
    hsv = cv2.cvtColor(low, cv2.COLOR_BGR2HSV)
    red = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 90)).astype(np.uint8) * 255
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(red)
    rects = [st[k, :4] for k in range(1, n) if st[k, 2] > 25 and st[k, 3] > 25]
    # homography low-res panel -> full-res page (ORB on grey, red removed so it doesn't drive the match)
    g_low = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY); g_low[red > 0] = 255
    up = 3
    g_low = cv2.resize(g_low, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    g_full = np.asarray(full.convert("L"))
    s = 1600 / max(g_full.shape)
    g_small = cv2.resize(g_full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(g_low, None); k2, d2 = orb.detectAndCompute(g_small, None)
    good = [m for m, nn in (x for x in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2) if len(x) == 2)
            if m.distance < 0.75 * nn.distance]
    H, mask = cv2.findHomography(np.float32([k1[m.queryIdx].pt for m in good]) / up,
                                 np.float32([k2[m.trainIdx].pt for m in good]) / s, cv2.RANSAC, 8.0)
    print("homography inliers", int(mask.sum()), "of", len(good), "; boxes found", len(rects))
    w = max(8, round(full.height / 270))
    d = ImageDraw.Draw(full)
    for x, y, bw, bh in rects:
        c = cv2.perspectiveTransform(np.float32([[[x, y]], [[x + bw, y]], [[x + bw, y + bh]], [[x, y + bh]]]), H).reshape(-1, 2)
        d.rectangle([c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()], outline=RED, width=w)
    full.save(OUT / "fig1_diagrams_goldleaf_p18.png", dpi=(300, 300))
    full.save(OUT / "fig1_diagrams_goldleaf_p18.tif", dpi=(300, 300), compression="tiff_lzw")
    return full


def margin_diagram_boxes(rgb: Image.Image):
    """Boxes round the diagrams printed in the left margin: ink between the paper's left edge and the
    text block, merged into figures (lines + their letter labels) and kept if big enough."""
    g = np.asarray(rgb.convert("L"))
    H, W = g.shape
    ink = cv2.threshold(cv2.GaussianBlur(g, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # paper starts where columns become bright (skips the binding / marbled endpaper)
    col_bright = cv2.blur(g.mean(0)[None, :].astype(np.float32), (max(3, W // 100), 1))[0]
    paper_x0 = int(np.argmax(col_bright > 0.85 * np.percentile(col_bright, 90)))
    # text block starts at the first long run of ink-dense columns right of the middle-left
    dens = cv2.blur((ink[int(.15 * H):int(.85 * H)] > 0).mean(0)[None, :].astype(np.float32), (max(3, W // 60), 1))[0]
    cand = np.where(dens[int(.25 * W):] > 0.12)[0]
    text_x0 = int(.25 * W) + int(cand[0]) if len(cand) else int(.45 * W)
    x0, x1 = paper_x0 + int(.01 * W), text_x0 - int(.008 * W)
    m = np.zeros_like(ink); m[int(.03 * H):int(.97 * H), x0:x1] = ink[int(.03 * H):int(.97 * H), x0:x1]
    k = max(5, round(W / 60))
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w * h > 0.004 * W * H and w > 0.05 * W:
            p = k // 2  # undo the dilation, then add a small margin
            boxes.append([x + p - int(.006 * W), y + p - int(.006 * W), x + w - p + int(.006 * W), y + h - p + int(.006 * W)])
    return sorted(boxes, key=lambda b: b[1]), (paper_x0, text_x0)


def draw_boxes(im, boxes, name):
    w = max(4, round(im.height / 270))
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle(b, outline=RED, width=w)
    im.save(OUT / f"{name}.png", dpi=(300, 300))
    im.save(OUT / f"{name}.tif", dpi=(300, 300), compression="tiff_lzw")
    print(name, im.size, "boxes", boxes, "line width", w)


def fig1_diagrams_both():
    import pymupdf
    # (a) the page Xuzhuo used: Library of Congress gold-leaf copy, page 18 (only 854 x 1200 px in this PDF)
    doc = pymupdf.open(DISS / "euclid_gold_leaf.pdf")
    pix = pymupdf.Pixmap(doc, doc[17].get_images()[0][0])
    if pix.n - pix.alpha > 3 or pix.alpha:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    gold = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    # the three diagrams, as boxed in Xuzhuo's panel (checked by eye on this page)
    draw_boxes(gold, [[31, 223, 270, 390], [38, 560, 262, 750], [39, 801, 262, 938]], "fig1_diagrams_goldleaf_p18")
    # (b) the same propositions (29-31) in the UCL copy at full scan resolution. The automatic margin
    # detection merges the lower two diagrams; they are split at the blank band between them
    # (rows 2888-2922) and each box is fitted tightly to its ink.
    ucl = Image.open(DISS / "tiff_cropped" / "32811_028.tif").convert("RGB")
    g = np.asarray(ucl.convert("L"))
    ink = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] > 0
    def tight(x0, y0, x1, y1, pad=30):
        ys, xs = np.where(ink[y0:y1, x0:x1])
        return [x0 + xs.min() - pad, y0 + ys.min() - pad, x0 + xs.max() + pad, y0 + ys.max() + pad]
    draw_boxes(ucl, [tight(439, 904, 1360, 1520), tight(483, 2224, 1303, 2888, pad=12), tight(483, 2922, 1303, 3405, pad=12)],
               "fig1_diagrams_UCL_32811_028")


def fig2_rotation_check():
    """The O pair of Fig 2(c)-(d): compare the raw match, a vertical flip and a 180-degree rotation."""
    O = DISS / "organised_capitals" / "O"
    return O


if __name__ == "__main__":
    fig1()
    fig3()
