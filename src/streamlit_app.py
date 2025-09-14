"""
Small shim for Hugging Face Spaces compatibility.

Some Spaces start Streamlit with `streamlit run src/streamlit_app.py`.
If that happens, this shim forwards execution to the real launcher
at the repository root (streamlit_app.py), which in turn runs src/app.py.
"""
from pathlib import Path
import runpy
import sys
import os

ROOT = Path(__file__).parent.parent.resolve()


print("shim: running src/streamlit_app.py -> delegating to ../streamlit_app.py")
sys.stdout.flush()


root_launcher = ROOT.joinpath("streamlit_app.py")
if root_launcher.exists():
    runpy.run_path(str(root_launcher), run_name="__main__")
else:
    
    app_script = ROOT.joinpath("src", "app.py")
    if not app_script.exists():
        raise FileNotFoundError(f"Neither {root_launcher} nor {app_script} exist")
    
    print("shim: root launcher missing — running src/app.py directly")
    sys.stdout.flush()
    runpy.run_path(str(app_script), run_name="__main__")
