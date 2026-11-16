"""A single background worker so long tasks (reading PDFs, detecting, grouping) never block the UI.
Jobs run one at a time, in order; the UI polls /api/jobs for progress."""
import threading
import time
import traceback
import uuid
from collections import deque

_jobs, _queue, _lock = {}, deque(), threading.Lock()
_wake = threading.Event()


def submit(kind, title, fn, **kw):
    jid = uuid.uuid4().hex[:8]
    job = dict(id=jid, kind=kind, title=title, status="queued", progress=0.0, message="Waiting…",
               created=time.time(), error=None, cancel=False, **kw)
    with _lock:
        _jobs[jid] = job
        _queue.append((jid, fn))
    _wake.set()
    return job


def cancel(jid):
    with _lock:
        if jid in _jobs:
            _jobs[jid]["cancel"] = True
            if _jobs[jid]["status"] == "queued":
                _jobs[jid]["status"] = "cancelled"


def listing():
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: -j["created"])
        return [dict(j) for j in jobs[:30]]


def busy():
    with _lock:
        return any(j["status"] in ("queued", "running") for j in _jobs.values())


def _worker():
    while True:
        _wake.wait(1.0)
        with _lock:
            item = _queue.popleft() if _queue else None
            if not _queue:
                _wake.clear()
        if not item:
            continue
        jid, fn = item
        job = _jobs[jid]
        if job["status"] == "cancelled":
            continue
        job["status"] = "running"

        def progress(f, msg):
            job["progress"] = float(min(max(f, 0), 1)); job["message"] = msg

        try:
            fn(progress=progress, cancel=lambda: job["cancel"])
            job["status"] = "cancelled" if job["cancel"] else "done"
            job["progress"] = 1.0 if job["status"] == "done" else job["progress"]
        except Exception as e:
            job["status"] = "failed"; job["error"] = f"{type(e).__name__}: {e}"
            job["message"] = "Failed: " + job["error"]
            traceback.print_exc()
        job["finished"] = time.time()


threading.Thread(target=_worker, daemon=True, name="initials-studio-jobs").start()
