#!/usr/bin/env bash
# Initials Studio - macOS / Linux launcher. First start installs everything (needs internet, ~5-10 min).
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Initials Studio for the first time (downloads ~300 MB)..."
  PY=$(command -v python3 || command -v python || true)
  if [ -z "$PY" ]; then echo "Python 3.10+ is needed: https://www.python.org/downloads/"; exit 1; fi
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m initials_studio "$@"
