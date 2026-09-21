"""
Using non-human glass brains
============================

Show how to use non human glass brain.
"""

import os
from pathlib import Path

import templateflow.api as tflow

import nilearn as ni
from nilearn.image import math_img
from nilearn.plotting import plot_glass_brain, show

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

# change global path to glass brain files
# different path resolution when running locally or in CI
if os.getenv("CI"):
    ni.plotting.GLASS_BRAIN_ASSETS = Path.cwd() / "glass_brain_files"
else:
    ni.plotting.GLASS_BRAIN_ASSETS = (
        Path(__file__).parent / "glass_brain_files"
    )

plot_glass_brain(
    brain_img,
    threshold=5000,
    black_bg=True,
    title="Waxholm Space atlas of the Sprague Dawley rat brain",
    alpha=1,
)

show()
