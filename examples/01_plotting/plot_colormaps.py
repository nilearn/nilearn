"""
Matplotlib colormaps in Nilearn
===============================

Visualize HCP connectome workbench color maps shipped with Nilearn
which can be used for plotting brain images on surface.

See :ref:`surface-plotting` for surface plotting details.
"""

import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting import show
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps


# %%
# Create a function to help us plot all the colormaps with their names.
#
def plot_color_gradients(color_maps):
    """Create figure and adjust figure height to number of colormaps.

    Adapted from the matplolib documentation.
    """
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    color_map_names = sorted(color_maps)

    nrows = len(color_map_names)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(
        top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99
    )

    for ax, name in zip(axs, color_map_names):
        ax.imshow(gradient, aspect="auto", cmap=name)
        ax.text(
            -0.01,
            0.5,
            name,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


# %%
# Plot matplotlib color maps
# --------------------------
plot_color_gradients(nilearn_cmaps)

# %%
# Plot matplotlib color maps
# --------------------------

deprecated_cmaps = ["Vega10", "Vega20", "Vega20b", "Vega20c", "spectral"]
m_cmaps = [
    m
    for m in plt.cm.datad
    if not m.endswith("_r") and m not in deprecated_cmaps
]

plot_color_gradients(m_cmaps)

show()

# sphinx_gallery_dummy_images=2
