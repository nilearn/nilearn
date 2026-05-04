"""Update code of the different vendored javascript libraries
based on the version listed in the package.json.
"""

import shutil
from pathlib import Path


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


SRC_FOLDER = root_dir() / "node_modules" / "@niivue" / "niivue"
DESTINATION_FILE = root_dir() / "nilearn" / "_assets" / "js" / "niivue.umd.js"

if not SRC_FOLDER.exists():
    raise RuntimeError("No 'node_modules' found. Please run 'npm install'.")

if (SRC_FOLDER / "dist" / "niivue.umd.js").exists():
    source_file = "node_modules/@niivue/niivue/dist/niivue.umd.js"

shutil.copyfile(source_file, DESTINATION_FILE)
