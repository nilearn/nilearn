"""
Matplotlib colormaps in Nilearn
===============================

Visualize HCP connectome workbench color maps shipped with Nilearn
which can be used for plotting brain images on surface.

See :ref:`surface-plotting` for surface plotting details.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting import show
from nilearn.plotting.cm import _cmap_d as nilearn_cmaps

try:
    matplotlib_maps = mpl.colormaps
except AttributeError:
    matplotlib_maps = plt.cm.datad

cmaps = {}

gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))


def plot_color_gradients(
    category, color_maps, color_map_names=None, sort=False
):
    """Create figure and adjust figure height to number of colormaps.

    Adapted from the matplolib documentation.
    """
    if color_map_names is None:
        color_map_names = color_maps.keys()
    # remove reversed maps
    color_map_names = remove_reversed_map(color_map_names)
    if sort:
        color_map_names = sorted(color_map_names)

    nrows = len(color_map_names)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
    fig.subplots_adjust(
        top=1 - 0.35 / figh, bottom=0.15 / figh, left=0.2, right=0.99
    )
    axs[0].set_title(f"{category} colormaps", fontsize=14)

    for ax, name in zip(axs, color_map_names):
        ax.imshow(gradient, aspect="auto", cmap=color_maps[name])
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


def remove_reversed_map(color_map_names):
    return [x for x in color_map_names if not x.endswith("_r")]


# %%
# Nilearn sequential color maps
# -----------------------------
# Sequential: change in lightness and often saturation of color incrementally,
# often using a single hue;
# should be used for representing information that has ordering.
#
# Compared to the perceptually uniform colormaps of matplotlib.
#
# See also:
# - https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential
# - https://matplotlib.org/stable/users/explain/colors/colormaps.html#sequential2
category = "Nilearn sequential"
color_map_names = [
    "black_blue",
    "black_purple",
    "black_pink",
    "black_red",
    "red_transparent",
    "green_transparent",
    "blue_transparent",
    "red_transparent_full_alpha_range",
    "green_transparent_full_alpha_range",
    "blue_transparent_full_alpha_range",
]
plot_color_gradients(
    category=category,
    color_maps=nilearn_cmaps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

category = "Matplotlib perceptually uniform sequential"
color_map_names = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
]
plot_color_gradients(
    category=category,
    color_maps=matplotlib_maps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

show()

# %%
# Nilearn diverging color maps
# ----------------------------
# Diverging: change in lightness
# and possibly saturation of two different colors
# that meet in the middle at an unsaturated color;
# should be used when the information being plotted
# has a critical middle value,
# such as topography or when the data deviates around zero.
#
# Also compared to those of matplotlib.

category = "Matplotlib diverging"
color_map_names = [
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
]
plot_color_gradients(
    category=category,
    color_maps=matplotlib_maps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

show()

# %%
# Nilearn cyclic color maps
# -------------------------
# Also compared to those of matplotlib.

category = "Nilearn cyclic"
color_map_names = [
    "cold_hot",
    "cold_white_hot",
    "hot_white_bone",
    "brown_blue",
    "hot_black_bone",
]
plot_color_gradients(
    category=category,
    color_maps=nilearn_cmaps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

category = "Matplotlib cyclic"
# hsv not included
color_map_names = ["twilight", "twilight_shifted"]
plot_color_gradients(
    category=category,
    color_maps=mpl.colormaps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

show()

# %%
# Nilearn other color maps
# ------------------------
category = "Nilearn other"
color_map_names = [
    "cyan_orange",
    "blue_red",
    "brown_cyan",
    "purple_green",
    "blue_orange",
    "cyan_copper",
]
plot_color_gradients(
    category=category,
    color_maps=nilearn_cmaps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

show()

# %%
# Nilearn misc color maps
# -----------------------

category = "Nilearn misc"
color_map_names = [
    "roy_big_bl",
    "videen_style",
    "ocean_hot",
]
plot_color_gradients(
    category=category,
    color_maps=nilearn_cmaps,
    color_map_names=color_map_names,
)
cmaps[category] = color_map_names

# show()

# %%
# Lightness of Nilearn colormaps
# ------------------------------
# Here we examine the lightness values of the matplotlib colormaps.
# Also compared to those of matplotlib.

from colorspacious import cspace_converter

mpl.rcParams.update({"font.size": 12})

# Number of colormap per subplot for particular cmap categories
_DSUBS = {
    "Nilearn sequential": 6,
    "Matplotlib perceptually uniform sequential": 5,
    "Nilearn diverging": 1,
    "Matplotlib diverging": 4,
    "Nilearn cyclic": 5,
    "Matplotlib cyclic": 2,
    "Nilearn other": 3,
    "Nilearn misc": 3,
}

# Spacing between the colormaps of a subplot
_DC = {
    "Nilearn sequential": 1.4,
    "Matplotlib perceptually uniform sequential": 1.4,
    "Nilearn diverging": 1.4,
    "Matplotlib diverging": 1.4,
    "Nilearn cyclic": 1.4,
    "Matplotlib cyclic": 1.4,
    "Nilearn other": 1.4,
    "Nilearn misc": 1.4,
}


# Indices to step through colormap
x = np.linspace(0.0, 1.0, 100)

# Do plot
for cmap_category, cmap_list in cmaps.items():
    # Do subplots so that colormaps have enough space.
    # Default is 6 colormaps per subplot.
    dsub = _DSUBS.get(cmap_category, 6)
    nsubplots = int(np.ceil(len(cmap_list) / dsub))

    # squeeze=False to handle similarly the case of a single subplot
    fig, axs = plt.subplots(
        nrows=nsubplots,
        squeeze=False,
        figsize=(7, 3 * nsubplots),
    )

    for i, ax in enumerate(axs.flat):
        locs = []  # locations for text labels

        for j, cmap in enumerate(cmap_list[i * dsub : (i + 1) * dsub]):
            # Get RGB values for colormap
            # and convert the colormap in # CAM02-UCS colorspace.
            # lab[0, :, 0] is the lightness.
            rgb = mpl.colormaps[cmap](x)[np.newaxis, :, :3]
            lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)

            # Plot colormap L values.
            # Do separately for each category so each plot can be pretty.
            # To make scatter markers change color along plot:
            # https://stackoverflow.com/q/8202605/

            y_ = lab[0, :, 0]
            c_ = x

            dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing
            ax.scatter(x + j * dc, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)

            # Store locations for colormap labels
            if cmap_category in (
                "Nilearn sequential",
                "Matplotlib perceptually uniform sequential",
                "Nilearn diverging",
                "Nilearn cyclic",
                "Matplotlib diverging",
            ):
                locs.append(x[-1] + j * dc)
            # elif cmap_category in (
            #     "Matplotlib diverging",
            #     "Nilearn diverging",
            # ):
            #     locs.append(x[int(x.size / 2.0)] + j * dc)

        # Set up the axis limits:
        #   * the 1st subplot is used as a reference for the x-axis limits
        #   * lightness values goes from 0 to 100 (y-axis limits)
        ax.set_xlim(axs[0, 0].get_xlim())
        ax.set_ylim(0.0, 100.0)

        # Set up labels for colormaps
        ax.xaxis.set_ticks_position("top")
        ticker = mpl.ticker.FixedLocator(locs)
        ax.xaxis.set_major_locator(ticker)
        formatter = mpl.ticker.FixedFormatter(
            cmap_list[i * dsub : (i + 1) * dsub]
        )
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=50)
        ax.set_ylabel("Lightness $L^*$", fontsize=12)

    ax.set_xlabel(f"{cmap_category} colormaps", fontsize=14)

    fig.tight_layout(h_pad=0.0, pad=1.5)
    plt.show()

# sphinx_gallery_dummy_images=2
