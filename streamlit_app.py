# streamlit_app.py  (place at repo root)
"""
Launcher for Hugging Face Spaces / Streamlit:
 - force XDG_CONFIG_HOME/HOME to a writable folder in the repo
 - create a minimal .streamlit/config.toml if missing
 - run the actual app (src/app.py) as __main__
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).parent.resolve()
STREAMLIT_DIR = ROOT.joinpath(".streamlit")

# --- FORCE environment variables (overwrite any existing HF defaults) ---
# Use a repo-local .streamlit directory so Streamlit won't try to write to '/'
os.environ["XDG_CONFIG_HOME"] = str(STREAMLIT_DIR)
# Ensure HOME is a writable folder inside the repo as well
os.environ["HOME"] = str(ROOT)
# Also set STREAMLIT_RUNTIME_DIR (added safety) and disable telemetry server writes
os.environ.setdefault("STREAMLIT_RUNTIME_DIR", str(ROOT.joinpath(".streamlit", "runtime")))

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

# Make sure runtime dir exists
runtime_dir = Path(os.environ.get("STREAMLIT_RUNTIME_DIR"))
runtime_dir.mkdir(parents=True, exist_ok=True)

# 3) Diagnostic print for logs (helps debug in Space logs)
print("streamlit_app launcher: XDG_CONFIG_HOME =", os.environ.get("XDG_CONFIG_HOME"))
print("streamlit_app launcher: HOME =", os.environ.get("HOME"))
print("streamlit_app launcher: STREAMLIT_RUNTIME_DIR =", os.environ.get("STREAMLIT_RUNTIME_DIR"))
sys.stdout.flush()

# 4) Run your real Streamlit app script (runs as __main__)
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
