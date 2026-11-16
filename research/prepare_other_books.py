"""Turn the non-test PDFs in the Diss folder into page images for a generalisation check.

- flowers_of_abu_mashar.pdf: rendered page by page (long side about 2000 px).
- judgement_of_the_stars.pdf: each page is a photo of an open book, rotated 90 degrees. It is
  turned upright and cut at the spine into a left and a right page.
(The music score already has split page images in Diss/music_score/pages.)

usage: python prepare_other_books.py
"""
import pymupdf
from PIL import Image

from common import DISS, HERE

out_root = HERE / "other_books"

d = pymupdf.open(DISS / "flowers_of_abu_mashar.pdf")
out = out_root / "flowers_of_abu_mashar"; out.mkdir(parents=True, exist_ok=True)
for i, p in enumerate(d):
    zoom = 2000 / max(p.rect.width, p.rect.height)
    pix = p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(out / f"page_{i}.jpg", quality=92)
print("flowers:", len(d), "pages")

d = pymupdf.open(DISS / "judgement_of_the_stars.pdf")
out = out_root / "judgement_of_the_stars"; out.mkdir(parents=True, exist_ok=True)
n = 0
for i, p in enumerate(d):
    pix = p.get_pixmap(dpi=72)  # the page is exactly one 1280x1800 scan, so 72 dpi = native pixels
    im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).rotate(-90, expand=True)
    w, h = im.size
    im.crop((0, 0, w // 2, h)).save(out / f"page_{i}_left.jpg", quality=92)
    im.crop((w // 2, 0, w, h)).save(out / f"page_{i}_right.jpg", quality=92)
    n += 2
print("judgement:", n, "half-spreads")
