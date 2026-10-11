"""Optional Streamlit entry point for the legacy dashboard.

Run from the repository root with:
    streamlit run legacy/streamlit/streamlit_app.py

The launcher adds the repository root for ``src.*`` imports and runs the
dashboard from its own directory so sample-data paths resolve consistently.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# This entry point lives at legacy/streamlit/streamlit_app.py; src.* modules
# and the app's sample data are rooted at the repository top level.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Run the actual app from its legacy dashboard directory.
_APP_PATH = _ROOT / "legacy" / "streamlit" / "app.py"
runpy.run_path(str(_APP_PATH), run_name="__main__")
