"""
streamlit_app.py — Streamlit Community Cloud entry point for ClaimGuard AI.

Streamlit Cloud requires the main file to be named streamlit_app.py at the
repo root. This file adds the repo root to sys.path and then runs app/app.py
via runpy so __file__ is set correctly and all relative imports resolve.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Repo root (where this file lives) must be on sys.path so src.* resolves
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Run the actual app
runpy.run_path(str(_ROOT / "app.py"), run_name="__main__")
