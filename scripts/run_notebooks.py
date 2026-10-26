"""
run_notebooks.py - execute the notebooks in order and save their outputs in place.

    python scripts/run_notebooks.py            # all notebooks
    python scripts/run_notebooks.py 03 04      # only notebooks whose name starts with 03 / 04
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

if __name__ == "__main__":
    prefixes = sys.argv[1:]
    for nb_path in sorted(NB_DIR.glob("0*.ipynb")):
        if prefixes and not any(nb_path.name.startswith(p) for p in prefixes):
            continue
        t = time.time()
        nb = nbformat.read(nb_path, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, nb_path)
        print(f"[ok] {nb_path.name}  ({time.time() - t:.0f}s)")
