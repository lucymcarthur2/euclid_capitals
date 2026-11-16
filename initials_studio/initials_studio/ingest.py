"""Turn uploaded PDFs / images into page images in a book folder."""
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PAGE_LONG_SIDE
from . import store

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
Image.MAX_IMAGE_PIXELS = 400_000_000  # big archival scans are fine


def _save_page(book, im: Image.Image, n: int, label: str):
    im = im.convert("RGB")
    if max(im.size) > PAGE_LONG_SIDE:
        im.thumbnail((PAGE_LONG_SIDE, PAGE_LONG_SIDE), Image.LANCZOS)
    pid = f"p{n:04d}"
    path = store.page_path(book["id"], pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=90)
    book["pages"].append(dict(id=pid, w=im.width, h=im.height, label=label))


def pdf_pages(path: Path):
    """Yield (PIL image, label) per PDF page: the embedded scan at native size when a page is a single
    image, otherwise the page rendered at 200 dpi."""
    import pymupdf
    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        imgs = page.get_images()
        im = None
        if len(imgs) == 1:
            try:
                pix = pymupdf.Pixmap(doc, imgs[0][0])
                if pix.n - pix.alpha > 3 or pix.alpha:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                mode = "RGB" if pix.n >= 3 else "L"
                im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                if min(im.size) < 400:  # tiny embedded image: render instead
                    im = None
            except Exception:
                im = None
        if im is None:
            z = 200 / 72
            pix = page.get_pixmap(matrix=pymupdf.Matrix(z, z))
            im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        yield im, f"{path.name} p.{i + 1}"


def add_files(book, files, progress=lambda f, msg: None):
    """files: list of Paths (PDFs and/or images), added to the book in the given order."""
    total = 0
    plan = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            import pymupdf
            n = pymupdf.open(f).page_count
        else:
            n = 1
        plan.append((f, n)); total += n
    done = len(book["pages"])
    k = 0
    for f, n in plan:
        if f.suffix.lower() == ".pdf":
            for im, label in pdf_pages(f):
                _save_page(book, im, done, label); done += 1; k += 1
                progress(k / max(total, 1), f"Reading {f.name}: page {k} of {total}")
        elif f.suffix.lower() in IMAGE_EXT:
            with Image.open(f) as im:
                _save_page(book, im, done, f.name)
            done += 1; k += 1
            progress(k / max(total, 1), f"Reading images: {k} of {total}")
    store.save_book(book)
    return book
