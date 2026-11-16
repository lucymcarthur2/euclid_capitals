"""Build Initials Studio's bundled examples (app/examples/) from this project's data.

- Crops-only corpus: the Ratdolt books (labelled capitals) + Santritter's Jandun (detections).
  Only the capital crops are shipped, not the page scans.
- Two full sample books with page images, openly licensed, detected by the app itself:
  Hyginus (Thomas de Blavis, Venice 1488; Université de Liège DONum, public domain) and
  Aquinas, De esse et essentia (Santritter & de Sanctis, Venice 1488; BSB Munich, Public Domain Mark).
- The woodblock grouping is precomputed by the app's own grouping code.

usage: python build_examples.py          (takes ~15-20 min: detection + matching)
"""
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "app" / "_examples_build"
os.environ["INITIALS_STUDIO_WORKSPACE"] = str(BUILD)
sys.path.insert(0, str(HERE / "app"))

from PIL import Image  # noqa: E402

from initials_studio import store  # noqa: E402
from initials_studio.extract import save_crop, run_detection  # noqa: E402
from initials_studio.grouping import regroup  # noqa: E402
from initials_studio.config import save_settings  # noqa: E402

import blocks_all  # noqa: E402  (the research script's book loaders)
from common import (DISS, load_pages, COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR, ARTES_IMG_DIR, ARTES_ANN_DIR,  # noqa: E402
                    PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR)

SAMPLE_PAGE_SIDE = 1800

CROPS_ONLY = [
    # key, title, printer, date, source, loader
    ("euclid", "Euclid, Elementa geometriae", "Erhard Ratdolt", "Venice, 1482", "Diss/jpegs_cropped (labelled)",
     lambda: blocks_all.gt_items("euclid", DISS / "jpegs_cropped", HERE / "annotations_corrected")),
    ("compilatio", "Leopold of Austria, Compilatio de astrorum scientia", "Erhard Ratdolt", "Augsburg, 1489",
     "Library of Congress scan (labelled)", lambda: blocks_all.gt_items("compilatio", COMPILATIO_IMG_DIR, COMPILATIO_ANN_DIR)),
    ("artes", "Artes orandi", "Erhard Ratdolt", "1485", "labelled", lambda: blocks_all.gt_items("artes", ARTES_IMG_DIR, ARTES_ANN_DIR)),
    ("ptolemy", "Ptolemy", "Erhard Ratdolt", "", "labelled", lambda: blocks_all.gt_items("ptolemy", PTOLEMY_IMG_DIR, PTOLEMY_ANN_DIR)),
    ("judgement", "Judgement of the Stars", "Erhard Ratdolt", "", "photographs of an open copy (reviewed labels)",
     blocks_all.judgement_items),
    ("jandun", "Johannes de Janduno, Quaestiones super libros Physicorum", "Johannes Lucilius Santritter & Hieronymus de Sanctis",
     "Venice, 1488", "BSB Munich bsb00054286, Public Domain Mark (detections >= 0.5)",
     lambda: blocks_all.det_items("jandun_santritter_1488", HERE / "other_books" / "jandun_santritter_1488")),
]
FULL = [
    ("Hyginus, Poeticon astronomicon", "Thomas de Blavis", "Venice, 1488",
     "Université de Liège, DONum hdl.handle.net/2268.1/7085 (public domain)", HERE / "other_books" / "hyginus_blavis_1488"),
    ("Thomas Aquinas, De esse et essentia", "Johannes Lucilius Santritter & Hieronymus de Sanctis", "Venice, 1488",
     "BSB Munich bsb00058317, Public Domain Mark", HERE / "other_books" / "aquinas_santritter_1488"),
]


def natural(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def build():
    if BUILD.exists():
        shutil.rmtree(BUILD)
    save_settings({})
    for key, title, printer, date, source, loader in CROPS_ONLY:
        items = loader()
        b = store.new_book(title, printer, date, source, example=True)
        b["crops_only"] = True
        caps, pages, open_page, im = [], {}, None, None
        for n, (_, pid, path, box) in enumerate(sorted(items, key=lambda it: (natural(str(it[1])), it[3][1]))):
            if open_page != path:
                im = Image.open(path); im.load(); open_page = path
            label = str(pid)
            page_id = "p" + "".join(ch if ch.isalnum() else "_" for ch in label)
            k = pages.setdefault(page_id, 0); pages[page_id] += 1
            cid = f"{page_id.lower()}_{k}"
            st = save_crop(b["id"], cid, im, box)
            caps.append(dict(id=cid, page=page_id, box=[float(v) for v in box], score=1.0, status="kept", stats=st))
        b["pages"] = [dict(id=p, w=0, h=0, label=p[1:]) for p in pages]
        b["n_pages"] = len(pages)
        b["status"] = "example"
        store.save_book(b); store.save_capitals(b["id"], caps)
        print(f"{title}: {len(caps)} capitals")

    for title, printer, date, source, folder in FULL:
        b = store.new_book(title, printer, date, source, example=True)
        files = sorted(folder.glob("page_*.jpg"), key=lambda p: natural(p.stem))
        for n, f in enumerate(files):
            with Image.open(f) as im:
                im = im.convert("RGB"); im.thumbnail((SAMPLE_PAGE_SIDE, SAMPLE_PAGE_SIDE), Image.LANCZOS)
                pid = f"p{n:04d}"
                dst = store.page_path(b["id"], pid); dst.parent.mkdir(parents=True, exist_ok=True)
                im.save(dst, quality=84)
                b["pages"].append(dict(id=pid, w=im.width, h=im.height, label=f"page {n + 1}"))
        store.save_book(b)
        run_detection(b, lambda f, m: print("\r" + m, end="", flush=True))
        b = store.get_book(b["id"]); b["example"] = True; store.save_book(b)
        print(f"\n{title}: {len(files)} pages")

    regroup(lambda f, m: print("\r" + m + " " * 10, end="", flush=True))
    print()
    out = HERE / "app" / "examples"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(BUILD / "books", out / "books")
    g = out / "grouping"; g.mkdir()
    shutil.copy(BUILD / "grouping" / "groups.json", g / "groups.json")
    for f in (BUILD / "grouping").glob("pairs_*.npz"):
        shutil.copy(f, g / f.name)
    shutil.copytree(BUILD / "grouping" / "features", g / "features")
    shutil.rmtree(BUILD)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"examples written to {out} ({size:.0f} MB)")


if __name__ == "__main__":
    build()
