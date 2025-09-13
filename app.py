# app.py  (place this at repo root and push)
"""
Top-level launcher for Hugging Face Spaces (Streamlit).
Ensures Streamlit config/metrics are written into repo .streamlit (writable),
then runs src/app.py as __main__.
"""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).parent.resolve()
STREAMLIT_DIR = ROOT.joinpath(".streamlit")

# 1) Prefer a repo-local config dir so Streamlit doesn't try to write to '/'
os.environ.setdefault("XDG_CONFIG_HOME", str(STREAMLIT_DIR))
# Some libs read HOME, ensure it's not '/'
os.environ.setdefault("HOME", str(ROOT))

# 2) Ensure .streamlit/config.toml exists with recommended settings
STREAMLIT_DIR.mkdir(parents=True, exist_ok=True)
cfg = STREAMLIT_DIR.joinpath("config.toml")
if not cfg.exists():
    cfg.write_text(
        "[server]\n"
        "headless = true\n"
        "port = 8501\n"
        "enableCORS = false\n"
        "enableWebsocketCompression = false\n\n"
        "[browser]\n"
        "gatherUsageStats = false\n",
        encoding="utf-8",
    )

# 3) Finally run your real app script
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
