"""Update code of the different vendored javascript libraries
based on the version listed in the package.json.
"""

import shutil
from pathlib import Path


def root_dir() -> Path:
    """Return path to root directory."""
    return Path(__file__).parent.parent


NODE_MODULES_DIR = root_dir() / "node_modules"

if not NODE_MODULES_DIR.exists():
    raise RuntimeError("No 'node_modules' found. Please run 'npm install'.")

ASSET_DIR = root_dir() / "nilearn" / "_assets" / "js"

source_file = (
    NODE_MODULES_DIR / "@niivue" / "niivue" / "dist" / "niivue.umd.js"
)
destination_file = ASSET_DIR / "niivue.umd.js"
shutil.copyfile(source_file, destination_file)

source_file = (
    NODE_MODULES_DIR / "plotly.js-gl3d-dist-min" / "plotly-gl3d.min.js"
)
destination_file = ASSET_DIR / "plotly-gl3d-latest.min.js"
shutil.copyfile(source_file, destination_file)

# TODO
# also adapt to update jquery, brainsprite
