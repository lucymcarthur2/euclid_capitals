"""Initials Studio web server (local only). Run with:  python -m initials_studio"""
import csv
import io
import json
import shutil
import time
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import STATIC_DIR, WORKSPACE, load_settings, save_settings, DEFAULTS
from . import store, jobs
from .extract import flag_book, FLAG_TEXT, add_manual, run_detection, visible
from .grouping import load_groups, regroup, group_flags, apply_edit, match_status
from .detector import model_info

app = FastAPI(title="Initials Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
THUMBS = WORKSPACE / "cache"
UPLOADS = WORKSPACE / "uploads"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- status / settings
@app.get("/api/status")
def status():
    return dict(model=model_info(), settings=load_settings(), defaults=DEFAULTS, busy=jobs.busy(),
                flag_text=FLAG_TEXT, workspace=str(WORKSPACE))


@app.post("/api/settings")
def set_settings(body: dict):
    return save_settings(body)


@app.get("/api/jobs")
def get_jobs():
    return jobs.listing()


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    jobs.cancel(jid)
    return {"ok": True}


# ---------------------------------------------------------------- books
def _book_summary(b, s, gflags):
    caps = flag_book(b, store.get_capitals(b["id"]), s,
                     {k.split("/", 1)[1]: v for k, v in gflags.items() if k.startswith(b["id"] + "/")})
    vis = [c for c in caps if c["visible"]]
    return dict(b, n_pages=len(b.get("pages", [])) or b.get("n_pages", 0), n_capitals=len(vis),
                n_flagged=sum(1 for c in vis if c["flags"] and c["status"] not in ("kept", "added")),
                n_rejected=sum(1 for c in caps if c["status"] == "rejected"),
                cover=next((c["id"] for c in vis), None), pages=None)


@app.get("/api/books")
def books():
    s, gf = load_settings(), group_flags()
    return [_book_summary(b, s, gf) for b in store.list_books()]


@app.post("/api/books")
async def upload(title: str = Form(""), printer: str = Form(""), date: str = Form(""),
                 one_book_per_pdf: bool = Form(True), files: List[UploadFile] = File(...)):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "upload").name
        dst = UPLOADS / f"{int(time.time() * 1000)}" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dst)
    pdfs = [p for p in saved if p.suffix.lower() == ".pdf"]
    imgs = [p for p in saved if p.suffix.lower() != ".pdf"]
    batches = []
    if one_book_per_pdf and len(pdfs) > 1:
        batches += [([p], p.stem) for p in pdfs]
    elif pdfs:
        batches.append((pdfs, title or pdfs[0].stem))
    if imgs:
        batches.append((imgs, title or "Uploaded images"))
    made = []
    for fs, t in batches:
        b = store.new_book(t.replace("_", " "), printer, date, source=", ".join(p.name for p in fs))
        made.append(b["id"])
        jobs.submit("import", f"Import “{b['title']}”", _import_job(b["id"], fs), book=b["id"])
    return {"books": made}


def _import_job(book_id, files):
    def run(progress, cancel):
        from .ingest import add_files
        b = store.get_book(book_id)
        b["status"] = "importing"; store.save_book(b)
        add_files(b, files, lambda f, m: progress(0.3 * f, m))
        for p in files:
            p.unlink(missing_ok=True)
            try:
                p.parent.rmdir()
            except OSError:
                pass
        b["status"] = "detecting"; store.save_book(b)
        run_detection(b, lambda f, m: progress(0.3 + 0.7 * f, m), cancel)
    return run


@app.patch("/api/books/{book_id}")
def edit_book(book_id: str, body: dict):
    b = store.get_book(book_id) or _404()
    for k in ("title", "printer", "date", "include_in_grouping"):
        if k in body:
            b[k] = body[k]
    store.save_book(b)
    return b


@app.delete("/api/books/{book_id}")
def delete_book(book_id: str):
    store.get_book(book_id) or _404()
    shutil.rmtree(store.book_dir(book_id), ignore_errors=True)
    return {"ok": True}


@app.post("/api/books/{book_id}/detect")
def redetect(book_id: str):
    b = store.get_book(book_id) or _404()
    if b.get("crops_only"):
        raise HTTPException(400, "This example has no page images to re-run detection on.")
    return jobs.submit("detect", f"Find capitals in “{b['title']}”",
                       lambda progress, cancel: run_detection(store.get_book(book_id), progress, cancel), book=book_id)


@app.get("/api/books/{book_id}")
def book_detail(book_id: str):
    b = store.get_book(book_id) or _404()
    s = load_settings()
    gf = {k.split("/", 1)[1]: v for k, v in group_flags().items() if k.startswith(book_id + "/")}
    caps = flag_book(b, store.get_capitals(book_id), s, gf)
    caps = [c for c in caps if c["score"] >= min(0.2, s["score_cutoff"]) or c["status"] in ("kept", "added")]
    return dict(book=b, capitals=caps)


@app.post("/api/books/{book_id}/capitals/{cap_id}")
def set_capital(book_id: str, cap_id: str, body: dict):
    caps = store.get_capitals(book_id)
    for c in caps:
        if c["id"] == cap_id:
            if body.get("status") in ("auto", "kept", "rejected"):
                c["status"] = body["status"] if not (c["status"] == "added" and body["status"] == "auto") else "added"
            if c["status"] == "rejected" and c["id"].split("_")[-1].startswith("m") and body.get("delete"):
                caps.remove(c)
            store.save_capitals(book_id, caps)
            return {"ok": True}
    _404()


@app.post("/api/books/{book_id}/capitals")
def add_capital(book_id: str, body: dict):
    store.get_book(book_id) or _404()
    return {"id": add_manual(book_id, body["page"], body["box"])}


@app.post("/api/books/{book_id}/bulk")
def bulk(book_id: str, body: dict):
    """status for many capitals at once: {"ids": [...], "status": "kept"|"rejected"|"auto"}"""
    ids = set(body.get("ids", []))
    caps = store.get_capitals(book_id)
    for c in caps:
        if c["id"] in ids and c["status"] != "added":
            c["status"] = body["status"]
        elif c["id"] in ids and body["status"] == "rejected":
            c["status"] = "rejected"
    store.save_capitals(book_id, caps)
    return {"ok": True}


# ---------------------------------------------------------------- images
def _thumb(src: Path, key: str, size: int, orient: int = 0):
    THUMBS.mkdir(parents=True, exist_ok=True)
    t = THUMBS / f"{key}_{size}_{orient}.jpg"
    if not t.exists() or t.stat().st_mtime < src.stat().st_mtime:
        with Image.open(src) as im:
            im = im.convert("RGB")
            if orient >= 4:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if orient % 4:
                im = im.rotate(-90 * (orient % 4), expand=True)
            im.thumbnail((size, size), Image.LANCZOS)
            im.save(t, quality=86)
    return FileResponse(t, media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


@app.get("/img/page/{book_id}/{page_id}")
def page_img(book_id: str, page_id: str, s: int = 1600):
    p = store.page_path(book_id, page_id)
    if not p.exists():
        _404()
    return _thumb(p, f"page_{book_id}_{page_id}", min(max(s, 200), 2400))


@app.get("/img/crop/{book_id}/{cap_id}")
def crop_img(book_id: str, cap_id: str, o: int = 0, s: int = 200):
    p = store.crop_path(book_id, cap_id)
    if not p.exists():
        _404()
    return _thumb(p, f"crop_{book_id}_{cap_id}", min(max(s, 60), 400), o % 8)


# ---------------------------------------------------------------- grouping
@app.post("/api/group")
def start_grouping(body: dict):
    rebuild = body.get("mode") == "rebuild"
    books = [b for b in body.get("books") or [] if store.get_book(b)] or None
    n = len(books) if books else len(store.list_books())
    return jobs.submit("group", f"Compare woodblocks in {n} book{'s' if n != 1 else ''}" + (" (from scratch)" if rebuild else ""),
                       lambda progress, cancel: regroup(progress, cancel, keep_edits=not rebuild, books=books))


@app.get("/api/groups")
def groups():
    g = load_groups()
    s = load_settings()
    books = {b["id"]: b for b in store.list_books()}
    caps = {}
    for bid in books:
        for c in store.get_capitals(bid):
            caps[f"{bid}/{c['id']}"] = c
    out = []
    for grp in g["groups"]:
        mem = []
        for m in grp["members"]:
            c = caps.get(m["key"])
            if c is None or not visible(c, s) or m["key"].split("/")[0] not in books:
                continue
            mem.append(dict(m, book=m["key"].split("/")[0], cap=m["key"].split("/")[1], page=c["page"]))
        if mem:
            out.append(dict(grp, members=mem))
    st = match_status(s)
    return dict(groups=out, books={k: dict(title=b["title"], printer=b.get("printer", ""), date=b.get("date", ""),
                                           example=b.get("example", False), **st["books"].get(k, {}))
                                   for k, b in books.items() if k in st["books"]},
                updated=g.get("updated"), params=g.get("params"), compared_pairs=st["compared_pairs"],
                mirror_pairs=g.get("mirror_pairs", []))


@app.post("/api/groups/edit")
def edit_groups(body: dict):
    apply_edit(body)
    return {"ok": True}


# ---------------------------------------------------------------- export
def _rows():
    s = load_settings()
    g = load_groups()
    where = {m["key"]: (grp["id"], m) for grp in g["groups"] for m in grp["members"]}
    from .matcher import ORIENT
    for b in store.list_books():
        for c in store.get_capitals(b["id"]):
            if not visible(c, s):
                continue
            gid, m = where.get(f"{b['id']}/{c['id']}", (None, {}))
            yield dict(book=b["title"], printer=b.get("printer", ""), date=b.get("date", ""), book_id=b["id"],
                       page=c["page"], page_label=next((p.get("label", "") for p in b.get("pages", []) if p["id"] == c["page"]), ""),
                       capital=c["id"], x1=c["box"][0], y1=c["box"][1], x2=c["box"][2], y2=c["box"][3],
                       score=c["score"], status=c["status"], pattern=gid,
                       printed_as=ORIENT[m.get("orient", 0)] if m else "", match_score=m.get("score", ""),
                       match_verdict=m.get("verdict", ""))


@app.get("/api/export/csv")
def export_csv():
    buf = io.StringIO()
    rows = list(_rows())
    w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["book"])
    w.writeheader(); w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="initials_studio_capitals.csv"'})


@app.get("/api/export/zip")
def export_zip():
    rows = list(_rows())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["book"]); w.writeheader(); w.writerows(rows)
        z.writestr("capitals.csv", out.getvalue())
        for r in rows:
            folder = f"pattern_{r['pattern']:04d}" if r["pattern"] is not None else "not_grouped"
            p = store.crop_path(r["book_id"], r["capital"])
            if p.exists():
                z.write(p, f"patterns/{folder}/{r['book_id']}__{r['capital']}.jpg")
        z.writestr("groups.json", json.dumps(load_groups(), indent=1))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="initials_studio_export.zip"'})


def _404():
    raise HTTPException(404, "not found")
