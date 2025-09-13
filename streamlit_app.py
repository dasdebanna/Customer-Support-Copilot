# streamlit_app.py  (place at repo root)
"""
Launcher for Hugging Face Spaces / Streamlit:
 - force XDG_CONFIG_HOME/HOME to a writable folder in the repo
 - create a minimal .streamlit/config.toml if missing
 - ensure Hugging Face / transformers cache dirs are repo-local and writable
 - add repo/src to sys.path so `from data_loader import ...` works
 - run the actual app (src/app.py) as __main__
"""
import os
from pathlib import Path
import runpy
import sys
import stat

ROOT = Path(__file__).parent.resolve()
STREAMLIT_DIR = ROOT.joinpath(".streamlit")

# --- FORCE environment variables (overwrite any existing HF defaults) ---
os.environ["XDG_CONFIG_HOME"] = str(STREAMLIT_DIR)
# Ensure HOME is a writable folder inside the repo as well
os.environ["HOME"] = str(ROOT)
# Also set STREAMLIT_RUNTIME_DIR (added safety)
os.environ.setdefault("STREAMLIT_RUNTIME_DIR", str(ROOT.joinpath(".streamlit", "runtime")))

# --- Setup repository-local cache for huggingface / transformers ---
# Put all caches under repo .cache so the process can create and write to them
CACHE_ROOT = ROOT.joinpath(".cache")
TF_CACHE = CACHE_ROOT.joinpath("transformers")
HF_HOME = CACHE_ROOT.joinpath("huggingface")
DATASETS_CACHE = CACHE_ROOT.joinpath("datasets")

# Set env vars used by HF libs
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TF_CACHE))
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASETS_CACHE))
# Optional: also set general cache-related vars
os.environ.setdefault("HF_METRICS_CACHE", str(CACHE_ROOT.joinpath("metrics")))

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

# Make sure cache dirs exist and are world-writable (so HF download locks work)
for d in (CACHE_ROOT, TF_CACHE, HF_HOME, DATASETS_CACHE, runtime_dir):
    try:
        d.mkdir(parents=True, exist_ok=True)
        # chmod 0o777 so non-root runtime user can create lock files & write
        d.chmod(0o777)
    except Exception:
        # best-effort; continue (we will show diagnostics below)
        pass

# --- ADD src directory to Python import path so app can import modules by name ---
SRC_DIR = str(ROOT.joinpath("src"))
if SRC_DIR not in sys.path:
    # Insert at front so local src overrides other packages with same names
    sys.path.insert(0, SRC_DIR)

# 3) Diagnostic prints for logs (helps debug in Space logs)
print("streamlit_app launcher: XDG_CONFIG_HOME =", os.environ.get("XDG_CONFIG_HOME"))
print("streamlit_app launcher: HOME =", os.environ.get("HOME"))
print("streamlit_app launcher: STREAMLIT_RUNTIME_DIR =", os.environ.get("STREAMLIT_RUNTIME_DIR"))
print("streamlit_app launcher: XDG_CACHE_HOME =", os.environ.get("XDG_CACHE_HOME"))
print("streamlit_app launcher: TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
print("streamlit_app launcher: HF_HOME =", os.environ.get("HF_HOME"))
print("streamlit_app launcher: HF_DATASETS_CACHE =", os.environ.get("HF_DATASETS_CACHE"))
print("streamlit_app launcher: sys.path[0] =", sys.path[0])
# show permission bits for the main cache dir
try:
    st = CACHE_ROOT.stat()
    print("streamlit_app launcher: .cache exists, mode:", oct(st.st_mode & 0o777))
except Exception:
    print("streamlit_app launcher: .cache stat failed or missing")
sys.stdout.flush()

# 4) Run your real Streamlit app script (runs as __main__)
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
