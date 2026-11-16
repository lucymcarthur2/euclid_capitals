"""Copy the built-in example books (and their precomputed woodblock grouping) into the workspace on
first start. Deleting an example in the app removes the copy only; it comes back with a new workspace."""
import shutil

from .config import EXAMPLES_DIR, BOOKS_DIR, GROUPING_DIR, WORKSPACE


def install_examples():
    marker = WORKSPACE / ".examples_installed"
    if marker.exists() or not EXAMPLES_DIR.exists():
        return
    BOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for d in sorted((EXAMPLES_DIR / "books").iterdir()):
        if d.is_dir() and not (BOOKS_DIR / d.name).exists():
            shutil.copytree(d, BOOKS_DIR / d.name)
    g = EXAMPLES_DIR / "grouping"
    if g.exists() and not GROUPING_DIR.exists():
        shutil.copytree(g, GROUPING_DIR)
    marker.write_text("ok")
