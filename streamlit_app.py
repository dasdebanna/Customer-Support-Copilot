# streamlit_app.py  (place at repo root)
"""
Launcher for Hugging Face Spaces / Streamlit:
 - pick a writable base dir (prefer repo root if writable, otherwise /tmp/<space>)
 - set XDG/STREAMLIT/TRANSFORMERS/HF cache env vars early (overwrite existing)
 - create the cache/runtime directories (best-effort) and attempt to chmod them
 - add repo/src to sys.path so `from data_loader import ...` works
 - run the real app (src/app.py) as __main__
"""
import os
import sys
import tempfile
from pathlib import Path
import runpy

ROOT = Path(__file__).parent.resolve()

def _writable_test(dir_path: Path) -> bool:
    """Return True if we can create & remove a small file in dir_path."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        test_file = dir_path.joinpath(".write_test")
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()
        return True
    except Exception:
        return False

# 1) Choose base writeable directory
# Prefer repo root if writable, otherwise fallback to /tmp/<repo-name>-<uid>
if _writable_test(ROOT):
    BASE = ROOT
else:
    # use /tmp which is generally writable in containers
    safe_suffix = ROOT.name.replace("/", "_").replace(" ", "_")
    BASE = Path(tempfile.gettempdir()).joinpath(f"{safe_suffix}_space_runtime")
    BASE.mkdir(parents=True, exist_ok=True)

# Helper to ensure and chmod (best-effort)
def ensure_dir(p: Path, mode: int = 0o777):
    try:
        p.mkdir(parents=True, exist_ok=True)
        # chmod might fail in some containers, ignore failures
        try:
            p.chmod(mode)
        except Exception:
            pass
    except Exception:
        pass

# 2) Setup paths (use BASE)
STREAMLIT_DIR = BASE.joinpath(".streamlit")
STREAMLIT_RUNTIME_DIR = STREAMLIT_DIR.joinpath("runtime")
CACHE_ROOT = BASE.joinpath(".cache")
TF_CACHE = CACHE_ROOT.joinpath("transformers")
HF_HOME = CACHE_ROOT.joinpath("huggingface")
DATASETS_CACHE = CACHE_ROOT.joinpath("datasets")
METRICS_CACHE = CACHE_ROOT.joinpath("metrics")

for p in (STREAMLIT_DIR, STREAMLIT_RUNTIME_DIR, CACHE_ROOT, TF_CACHE, HF_HOME, DATASETS_CACHE, METRICS_CACHE):
    ensure_dir(p)

# 3) Force environment variables (overwrite any existing ones)
# Streamlit config and runtime
os.environ["XDG_CONFIG_HOME"] = str(STREAMLIT_DIR)
os.environ["STREAMLIT_RUNTIME_DIR"] = str(STREAMLIT_RUNTIME_DIR)
# Set HOME to BASE so some libs compute caches relative to it (safe)
os.environ["HOME"] = str(BASE)

# Hugging Face / transformers caches (overwrite)
os.environ["XDG_CACHE_HOME"] = str(CACHE_ROOT)
os.environ["TRANSFORMERS_CACHE"] = str(TF_CACHE)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_DATASETS_CACHE"] = str(DATASETS_CACHE)
os.environ.setdefault("HF_METRICS_CACHE", str(METRICS_CACHE))

# 4) Add src dir to sys.path so imports like `from data_loader import ...` work
SRC_DIR = str(ROOT.joinpath("src"))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 5) Diagnostics (will appear in HF container logs)
def _stat_mode(p: Path):
    try:
        return oct(p.stat().st_mode & 0o777)
    except Exception:
        return "MISSING"

print("streamlit_app launcher: BASE =", str(BASE))
print("streamlit_app launcher: XDG_CONFIG_HOME =", os.environ.get("XDG_CONFIG_HOME"))
print("streamlit_app launcher: STREAMLIT_RUNTIME_DIR =", os.environ.get("STREAMLIT_RUNTIME_DIR"))
print("streamlit_app launcher: HOME =", os.environ.get("HOME"))
print("streamlit_app launcher: XDG_CACHE_HOME =", os.environ.get("XDG_CACHE_HOME"))
print("streamlit_app launcher: TRANSFORMERS_CACHE =", os.environ.get("TRANSFORMERS_CACHE"))
print("streamlit_app launcher: HF_HOME =", os.environ.get("HF_HOME"))
print("streamlit_app launcher: HF_DATASETS_CACHE =", os.environ.get("HF_DATASETS_CACHE"))
print("streamlit_app launcher: sys.path[0] =", sys.path[0])
print(".streamlit mode:", _stat_mode(STREAMLIT_DIR))
print("runtime mode:", _stat_mode(STREAMLIT_RUNTIME_DIR))
print(".cache mode:", _stat_mode(CACHE_ROOT))
print("transformers cache mode:", _stat_mode(TF_CACHE))
print("hf home mode:", _stat_mode(HF_HOME))
sys.stdout.flush()

# 6) Create a minimal config.toml in chosen .streamlit (if not present)
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

# 7) Finally run the app script (as if python src/app.py)
runpy.run_path(str(ROOT.joinpath("src", "app.py")), run_name="__main__")
