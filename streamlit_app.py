# streamlit_app.py
"""
Wrapper so Hugging Face Spaces (Streamlit SDK) can launch the app.
It simply runs src/app.py as if it were the main file.
"""

from pathlib import Path
import runpy

# Run src/app.py as the main script
runpy.run_path(str(Path(__file__).parent.joinpath("src", "app.py")), run_name="__main__")
