# streamlit_app.py  (place at repo root)
"""
Launcher for Hugging Face Spaces / Streamlit:
 - ensure Streamlit config/metrics are written to a writable location
 - create a minimal .streamlit/config.toml if missing
 - run the actual app (src/app.py) as __main__
"""
import os
from pathlib import Path
import runpy

ROOT = Path(__file__).parent.resolve()
STREAMLIT_DIR = ROOT.joinpath(".streamlit")

# 1) Ensure container will not attempt to write to '/' for Streamlit config or metrics
# Set XDG_CONFIG_HOME to repo .streamlit unless the user already set one
os.environ.setdefault("XDG_CONFIG_HOME", str(STREAMLIT_DIR))
# Ensure HOME is not root; some environments expect HOME for user files
os.environ.setdefault("HOME", str(ROOT))

# 2) Create .streamlit and a minimal config.toml if it doesn't exist
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

# 3) Run your real Streamlit app script (runs as __main__)
# Use runpy so src/app.py will behave the same way as running `python src/app.py`
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
