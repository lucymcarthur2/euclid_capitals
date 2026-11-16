"""Download every page of a digitised book from a IIIF manifest (Munich MDZ, and most big libraries)
as page_<n>.jpg, long side MAX px. Pages already on disk are skipped, so it can be re-run.

usage: python fetch_iiif.py MANIFEST_URL_or_BSB_ID OUT_NAME [--max 2000]
   e.g. python fetch_iiif.py bsb00045897 avienus_strata_1488
Writes other_books/OUT_NAME/ and other_books/OUT_NAME/source.txt (manifest URL + title + licence).
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "Mozilla/5.0 (research script; capital detection)"}


def get(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


ap = argparse.ArgumentParser()
ap.add_argument("manifest"); ap.add_argument("name"); ap.add_argument("--max", type=int, default=2000)
a = ap.parse_args()
url = a.manifest if a.manifest.startswith("http") else \
    f"https://api.digitale-sammlungen.de/iiif/presentation/v2/{a.manifest}/manifest"
m = json.loads(get(url))
out = HERE / "other_books" / a.name
out.mkdir(parents=True, exist_ok=True)
canvases = m["sequences"][0]["canvases"]
(out / "source.txt").write_text(f"{url}\n{m.get('label')}\n{m.get('license', m.get('attribution', ''))}\n",
                                encoding="utf-8")
print(m.get("label"), "-", len(canvases), "pages", flush=True)
for i, c in enumerate(canvases):
    f = out / f"page_{i}.jpg"
    if f.exists() and f.stat().st_size > 10000:
        continue
    svc = c["images"][0]["resource"]["service"]["@id"].rstrip("/")
    f.write_bytes(get(f"{svc}/full/!{a.max},{a.max}/0/default.jpg"))
    if i % 25 == 0:
        print(f"{i + 1}/{len(canvases)}", flush=True)
    time.sleep(0.3)  # be polite to the library's server
print("done", out)
