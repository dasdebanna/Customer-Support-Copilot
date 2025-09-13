# streamlit_app.py  (place at repo root)
"""
Launcher for Hugging Face Spaces / Streamlit:
 - force XDG_CONFIG_HOME/HOME to a writable folder in the repo
 - create a minimal .streamlit/config.toml if missing
 - ensure Hugging Face / transformers cache dirs are repo-local and writable
 - add repo/src to sys.path so imports like `from data_loader import ...` work
 - run the actual app (src/app.py) as __main__
Notes:
 - We *assign* env vars (not setdefault) to ensure they are effective even if
   the environment was pre-populated earlier by the runtime.
 - We chmod cache dirs to 0o777 so non-root container user can create locks/write.
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).parent.resolve()
STREAMLIT_DIR = ROOT.joinpath(".streamlit")

# ------------------------------
# Force environment variables (overwrite any existing HF / Streamlit defaults)
# ------------------------------
# Use a repo-local .streamlit directory so Streamlit won't try to write to '/'
os.environ["XDG_CONFIG_HOME"] = str(STREAMLIT_DIR)
# Ensure HOME is a writable folder inside the repo as well
os.environ["HOME"] = str(ROOT)
# Force STREAMLIT_RUNTIME_DIR to repo-local runtime (overwrite anything)
os.environ["STREAMLIT_RUNTIME_DIR"] = str(ROOT.joinpath(".streamlit", "runtime"))

# Set repo-local caches for HF libs and transformers (overwrite any existing)
CACHE_ROOT = ROOT.joinpath(".cache")
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT)
# transformers prefers TRANSFORMERS_CACHE (deprecated) or HF_HOME; set both for compatibility
TF_CACHE = CACHE_ROOT.joinpath("transformers")
os.environ["TRANSFORMERS_CACHE"] = str(TF_CACHE)
HF_HOME = CACHE_ROOT.joinpath("huggingface")
os.environ["HF_HOME"] = str(HF_HOME)
DATASETS_CACHE = CACHE_ROOT.joinpath("datasets")
os.environ["HF_DATASETS_CACHE"] = str(DATASETS_CACHE)
# optional metric cache
os.environ.setdefault("HF_METRICS_CACHE", str(CACHE_ROOT.joinpath("metrics")))

# ------------------------------
# Create config / runtime / cache directories and make them writable
# ------------------------------
def ensure_dir_and_chmod(p: Path, mode: int = 0o777):
    try:
        p.mkdir(parents=True, exist_ok=True)
        p.chmod(mode)
    except Exception as e:
        # best-effort; continue and print diagnostics
        print(f"streamlit_app launcher: ensure_dir_and_chmod failed for {p}: {e}")

# Create important directories
ensure_dir_and_chmod(STREAMLIT_DIR)
ensure_dir_and_chmod(ROOT.joinpath(".streamlit", "runtime"))
ensure_dir_and_chmod(CACHE_ROOT)
ensure_dir_and_chmod(TF_CACHE)
ensure_dir_and_chmod(HF_HOME)
ensure_dir_and_chmod(DATASETS_CACHE)
ensure_dir_and_chmod(Path(os.environ.get("STREAMLIT_RUNTIME_DIR")))

# ------------------------------
# Create a minimal streamlit config if missing
# ------------------------------
cfg = STREAMLIT_DIR.joinpath("config.toml")
if not cfg.exists():
    try:
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
    except Exception as e:
        print("streamlit_app launcher: failed to write config.toml:", e)

# ------------------------------
# Add src directory to Python import path so app can import modules by name
# ------------------------------
SRC_DIR = str(ROOT.joinpath("src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# ------------------------------
# Print diagnostics (critical - will show in HF container logs)
# ------------------------------
def stat_mode(p: Path):
    try:
        s = p.stat()
        return oct(s.st_mode & 0o777)
    except Exception:
        return "MISSING"

print("streamlit_app launcher: XDG_CONFIG_HOME =", os.environ.get("XDG_CONFIG_HOME"))
print("streamlit_app launcher: HOME =", os.environ.get("HOME"))
print("streamlit_app launcher: STREAMLIT_RUNTIME_DIR =", os.environ.get("STREAMLIT_RUNTIME_DIR"))
print("streamlit_app launcher: XDG_CACHE_HOME =", os.environ.get("XDG_CACHE_HOME"))
print("streamlit_app launcher: TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
print("streamlit_app launcher: HF_HOME =", os.environ.get("HF_HOME"))
print("streamlit_app launcher: HF_DATASETS_CACHE =", os.environ.get("HF_DATASETS_CACHE"))
print("streamlit_app launcher: sys.path[0] =", sys.path[0])
print(".cache dir mode:", stat_mode(CACHE_ROOT))
print("transformers cache mode:", stat_mode(TF_CACHE))
print("hf home mode:", stat_mode(HF_HOME))
print("datasets cache mode:", stat_mode(DATASETS_CACHE))
sys.stdout.flush()

# ------------------------------
# Run the real app
# ------------------------------
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
