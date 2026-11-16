"""Paths and default parameters for Initials Studio."""
import json
import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):  # packaged as "Initials Studio.exe": data sits next to the exe
    APP_DIR = Path(sys.executable).resolve().parent
    STATIC_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "initials_studio" / "static"
else:
    APP_DIR = Path(__file__).resolve().parent.parent      # the shareable folder
    STATIC_DIR = Path(__file__).resolve().parent / "static"
PKG_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
EXAMPLES_DIR = APP_DIR / "examples"
WORKSPACE = Path(os.environ.get("INITIALS_STUDIO_WORKSPACE", APP_DIR / "workspace"))
BOOKS_DIR = WORKSPACE / "books"
GROUPING_DIR = WORKSPACE / "grouping"
SETTINGS_FILE = WORKSPACE / "settings.json"

PAGE_LONG_SIDE = 2400   # page images are stored at most this large
CROP_LONG_SIDE = 400    # saved capital crops
DETECT_FLOOR = 0.05     # everything above this is kept internally, so the cut-off can change instantly

DEFAULTS = {
    # detection
    "score_cutoff": 0.5,          # boxes below this are hidden (0.94 = strict, Euclid-tuned)
    "flag_low_confidence": 0.9,   # kept boxes below this get a "low confidence" flag
    "flag_size_factor": 2.2,      # flag boxes this many times bigger/smaller than the book's typical capital
    # grouping
    "same_block": 0.70,           # average match score to join a pattern (calibrated on Euclid)
    "unclear_band": 0.50,         # scores between this and same_block are flagged "unclear"
    "allow_rotations": True,      # blocks printed turned 90/180/270 degrees
    "neighbours": 60,             # candidates per capital fully compared (speed vs. recall)
    "workers": max(1, (os.cpu_count() or 2) - 1),
}


def load_settings() -> dict:
    s = dict(DEFAULTS)
    if SETTINGS_FILE.exists():
        try:
            s.update({k: v for k, v in json.loads(SETTINGS_FILE.read_text()).items() if k in DEFAULTS})
        except Exception:
            pass
    return s


def save_settings(new: dict) -> dict:
    s = load_settings()
    for k, v in new.items():
        if k in DEFAULTS:
            s[k] = type(DEFAULTS[k])(v)
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(s, indent=1))
    return s
