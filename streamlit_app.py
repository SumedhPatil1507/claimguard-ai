"""
streamlit_app.py — Streamlit Community Cloud entry point for ClaimGuard AI.

This file simply re-exports the full app from app/app.py.
Streamlit Cloud requires the main file to be at the repo root named streamlit_app.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path so all src.* imports resolve
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Run the actual app module — exec'ing it here makes Streamlit treat
# all the st.* calls as if they came from this file directly.
_app_path = _ROOT / "app" / "app.py"
exec(compile(_app_path.read_text(encoding="utf-8"), str(_app_path), "exec"))
