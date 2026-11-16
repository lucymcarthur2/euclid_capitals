"""Workspace storage: one folder per book, plain JSON + JPEG files (easy to back up or inspect).

books/<book_id>/book.json         title, printer, date, source, pages, status
books/<book_id>/pages/<page>.jpg  page images (absent for crops-only example books)
books/<book_id>/capitals.json     every detection: box, score, flags, status (auto/kept/rejected/added)
books/<book_id>/crops/<cap>.jpg   one image per capital (used for review and matching)
"""
import json
import re
import threading
import time
import uuid
from pathlib import Path

from .config import BOOKS_DIR

_lock = threading.RLock()


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "book"
    return f"{s}_{uuid.uuid4().hex[:6]}"


def book_dir(book_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", book_id):
        raise ValueError("bad book id")
    return BOOKS_DIR / book_id


def read_json(path: Path, default):
    with _lock:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data):
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)


def list_books():
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for d in sorted(BOOKS_DIR.iterdir()):
        b = read_json(d / "book.json", None)
        if b:
            out.append(b)
    return sorted(out, key=lambda b: (not b.get("example", False), b.get("created", 0)))


def get_book(book_id):
    return read_json(book_dir(book_id) / "book.json", None)


def save_book(book):
    write_json(book_dir(book["id"]) / "book.json", book)


def new_book(title, printer="", date="", source="", example=False):
    b = dict(id=slug(title), title=title, printer=printer, date=date, source=source, example=example,
             pages=[], status="new", created=time.time(), crops_only=False)
    save_book(b)
    return b


def get_capitals(book_id):
    return read_json(book_dir(book_id) / "capitals.json", [])


def save_capitals(book_id, caps):
    write_json(book_dir(book_id) / "capitals.json", caps)


def crop_path(book_id, cap_id) -> Path:
    if not re.fullmatch(r"[a-z0-9_]+", cap_id):
        raise ValueError("bad capital id")
    return book_dir(book_id) / "crops" / f"{cap_id}.jpg"


def page_path(book_id, page_id) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_\-]+", page_id):
        raise ValueError("bad page id")
    return book_dir(book_id) / "pages" / f"{page_id}.jpg"
