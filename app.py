"""Streamlit Community Cloud Root Entrypoint.

Delegates execution to app/Home.py so that any main file path setting
('app.py', 'streamlit_app.py', or 'app/Home.py') runs the multi-page app successfully.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

# Ensure src/ is on sys.path
root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

home_page = root_dir / "app" / "Home.py"
runpy.run_path(str(home_page), run_name="__main__")
