#!/bin/env python3

from nilearn import plotting
from nilearn import datasets
import numpy as np

# get some coords
power = datasets.fetch_coords_power_2011()
coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T

np.random.seed(0)

# node strength
node_strength = np.random.rand(264,)
node_sizes = np.random.rand(264,)*50

brain = plotting.plot_markers(
    node_strength,
    coords,
    node_vmin=0,
    node_vmax=1,
    node_size=node_sizes,
    node_cmap = "rainbow",
    display_mode="lyrz"
)

