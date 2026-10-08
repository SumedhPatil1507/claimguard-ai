"""
streamlit_app.py — Top-level Streamlit launcher for ClaimGuard AI.

Run directly via:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Ensure root directory is on sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Launch the Streamlit dashboard
_APP_PATH = _ROOT / "legacy" / "streamlit" / "app.py"
runpy.run_path(str(_APP_PATH), run_name="__main__")
