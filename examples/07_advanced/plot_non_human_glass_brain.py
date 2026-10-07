"""
Using non-human glass brains
============================

Show how to use non human glass brain.

The brain contours drawn by :func:`~nilearn.plotting.plot_glass_brain`
are stored as JSON files.
Here we start from brain contours drawn as SVG files
(for example with `Inkscape <https://inkscape.org/>`_),
convert them to the JSON format,
and use them to plot the Waxholm Space atlas of the Sprague Dawley rat brain.
"""

# %%
# Convert the SVG files to JSON
# -----------------------------
# The conversion is done by the ``svg_to_json`` function
# of ``maint_tools/svg_to_json_converter.py``,
# also available as a command line script.
# It only supports SVG paths that are made of
# (relative or absolute) moves, lines, cubic Bézier curves and closed paths.
# Paths without a stroke are invisible and are ignored.
import os
import sys
import tempfile
from pathlib import Path

import templateflow.api as tflow

import nilearn as ni
from nilearn.image import math_img
from nilearn.plotting import plot_glass_brain, show

# different path resolution when running locally or in CI
example_folder = Path.cwd() if os.getenv("CI") else Path(__file__).parent

sys.path.append(str(example_folder.parents[1] / "maint_tools"))
from fit_glass_brain_transform import fit_transforms, write_transforms
from svg_to_json_converter import svg_to_json

svg_folder = example_folder / "glass_brain_files"

# %%
# The glass brain needs a view for each of the 3 directions
# (``x``: side, ``y``: back, ``z``: top).
# Here the front and back views share the same contour.
svg_files = {
    "side": "brain_schematics_side.svg",
    "back": "brain_schematics_front.svg",
    "top": "brain_schematics_top.svg",
}

# %%
# Fetch the template
# ------------------
template = "WHS"

fetched_files = tflow.get(template, resolution=2, suffix="T2star")
brain_mask = tflow.get(
    template, resolution=2, desc="brain", suffix="mask", atlas=None
)

# The T2star template is not skull-stripped:
# keep only the brain so that it fits the brain contours.
brain_img = math_img(
    "img * mask",
    img=str(fetched_files),
    mask=str(brain_mask),
)

# %%
# Convert the SVG files to JSON
# -----------------------------
# We generate the JSON files in a temporary folder.
json_folder = tempfile.TemporaryDirectory()

for view, svg_file in svg_files.items():
    svg_to_json(
        svg_folder / svg_file,
        Path(json_folder.name) / f"brain_schematics_{view}.json",
    )

# %%
# Align the contours on the template
# ----------------------------------
# The contours are drawn in arbitrary units:
# we also need to tell how to align them on the template.
# The transform of each view is found by maximizing the overlap
# between the contour and the silhouette of the brain mask of the template
# (see ``maint_tools/fit_glass_brain_transform.py``,
# also available as a command line script).
# This takes several seconds.
transforms = fit_transforms(json_folder.name, brain_mask, verbose=True)

# store the transforms with the contours
# so they are used when plotting the glass brain
write_transforms(json_folder.name, transforms)

# %%
# Plot the template on the glass brain
# ------------------------------------
# We change the global path to the glass brain files
# for the time of the plot.
default_glass_brain_assets = ni.plotting.GLASS_BRAIN_ASSETS
ni.plotting.GLASS_BRAIN_ASSETS = Path(json_folder.name)

plot_glass_brain(
    brain_img,
    threshold=5000,
    black_bg=True,
    title="Waxholm Space atlas of the Sprague Dawley rat brain",
    alpha=1,
)

ni.plotting.GLASS_BRAIN_ASSETS = default_glass_brain_assets
json_folder.cleanup()

show()
