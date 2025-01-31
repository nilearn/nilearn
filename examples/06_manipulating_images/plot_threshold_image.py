# %%
"""
Image thresholding
==================

The goal of this example is to illustrate the use of the function
:func:`~nilearn.image.threshold_img` together with ``threshold`` and
``two_sided`` parameters to view certain values in image data.

The ``threshold`` parameter can take both positive and negative values.
``two_sided`` parameter is complementary to ``threshold`` effecting its
behavior.

"""

# %%
# Image without threshold
# -----------------------
#
# We will first load the dataset and display the image without manipulation.

from nilearn import datasets, plotting

cmap = "RdBu_r"
cut_coords = [-26, -33, 59]

image = datasets.load_sample_motor_activation_image()

plotting.plot_stat_map(
    image,
    colorbar=True,
    cmap=cmap,
    title="image without threshold",
    cut_coords=cut_coords,
)

# %%
# Image thresholded at 2 when two_sided=True
# ------------------------------------------
#
# Now we will use ``threshold=2`` together with ``two_sided=True`` to
# threshold the image. When ``two_sided=True``, we can only use positive
# values for ``threshold``.
#
# This will set all image values between -2 and 2 to 0.

import matplotlib.pyplot as plt

from nilearn.image import threshold_img

thresholded_img = threshold_img(
    image,
    threshold=2,
    cluster_threshold=0,
    two_sided=True,
    copy=True,
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(8, 8),
)

plotting.plot_stat_map(
    image,
    colorbar=True,
    cmap=cmap,
    title="image without threshold",
    axes=axes[0],
    cut_coords=cut_coords,
)

plotting.plot_stat_map(
    thresholded_img,
    colorbar=True,
    cmap=cmap,
    title="image thresholded at 2 with two_sided=True",
    cut_coords=cut_coords,
    axes=axes[1],
)

# %%
# Image thresholded at 2 when two_sided=False
# -------------------------------------------
#
# Now we will use ``threshold=2`` together with ``two_sided=False`` to
# see the effect.
#
# This will set all image values below 2 to 0.

import matplotlib.pyplot as plt

from nilearn.image import threshold_img

thresholded_img = threshold_img(
    image,
    threshold=2,
    cluster_threshold=0,
    two_sided=False,
    copy=True,
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(8, 8),
)

plotting.plot_stat_map(
    image,
    colorbar=True,
    cmap=cmap,
    title="image without threshold",
    axes=axes[0],
    cut_coords=cut_coords,
)

plotting.plot_stat_map(
    thresholded_img,
    colorbar=True,
    cmap=cmap,
    title="image thresholded at 2 with two_sided=False",
    cut_coords=cut_coords,
    axes=axes[1],
)

# %%
# Image thresholded at -2 when two_sided=False
# --------------------------------------------
#
# Now we will use ``threshold=-2`` together with ``two_sided=False`` to
# see the effect.
#
# This will set all image values above -2 to 0.

import matplotlib.pyplot as plt

from nilearn.image import threshold_img

thresholded_img = threshold_img(
    image,
    threshold=-2,
    cluster_threshold=0,
    two_sided=False,
    copy=True,
)


fig, axes = plt.subplots(
    2,
    1,
    figsize=(8, 8),
)

plotting.plot_stat_map(
    image,
    colorbar=True,
    cmap=cmap,
    title="image without threshold",
    axes=axes[0],
    cut_coords=cut_coords,
)

plotting.plot_stat_map(
    thresholded_img,
    colorbar=True,
    cmap=cmap,
    title="image thresholded at -2 with two_sided=False",
    cut_coords=cut_coords,
    axes=axes[1],
)
# -
