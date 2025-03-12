r"""
Plotting images with transparent thresholding
=============================================

Standard thresholding means that any data below a threshold value
is completely hidden from view.
However, "transparent thresholding" allows
for the same suprathreshold results to be observed,
while also showing subthreshold information
with an opacity that fades with decreasing magnitude
(:footcite:t:`Allen2012`, :footcite:t:`Chen2022`,
:footcite:t:`Taylor2023`, :footcite:t:`Sundermann2024`).

This makes use of the "alpha" value that overlays can have
when using some plotting functions (like :func:`matplotlib.pyplot.imshow`).
This "alpha" value goes from 0 (perfectly transparent) to 1 (perfectly opaque).

Consider having an underlay color :math:`RGB_{U}`
and overlay color :math:`RGB_{O}`,
where a threshold value T is applied
and we consider how an element (voxel, node...) with value M is shown.

"Opaque" thresholding
---------------------

if :math:`\lvert M \lvert >= T : alpha=1`,
meaning the overlay is shown as: :math:`RGB_{O}`
else : :math:`alpha=0`, meaning the underlay is shown as: :math:`RGB_{U}`

"Transparent" thresholding
--------------------------

The steepness of fading can be linear or quadratic. Linear is shown below.

If :math:`\lvert M \lvert >= T : alpha=1`,
meaning the overlay is shown as: :math:`RGB_{O}`.

Otherwise :math:`alpha = (\lvert M \lvert /  T)`,
merging :math:`RGB_{O}` and :math:`RGB_{U}` as:

:math:`RGB_{final} = (1-alpha) * RGB_{U} + alpha * RGB_{O}`.

In the end, this is just a small tweak for the case of subthreshold data.
In that case, alpha is nonzero (rather than simply 0).

Additionally, a contour can be placed around the suprathreshold regions,
to further highlight them.

So, the differences between standard, opaque thresholding
and the suggested transparent thresholding
is shown in the rest of this example.

..  admonition:: Benefits
    :class: important

    Implementing transparent thresholding can help provide
    more informative results reporting and
    more accurate interpretations,
    can facilitate quality control
    and improve reproducibility checks.

"""

# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.plotting import plot_stat_map, show

# %%
# Load the image we will use to demonstrate.
#

image = datasets.load_sample_motor_activation_image()

# %%
# Let's use some slightly different plotting parameters
# that should work better with transparency plotting.
# For example, let's pick a diverging colormap
# that diverges from black and not from white
# as the default colormap does.
#
# .. TODO switch to ``berlin`` color map when bumping matplotlib
#    as it is not a cyclic color map like ``cold_hot``.
#

vmin = 0.5
threshold = 3
figure_width = 8

plotting_config = {
    "display_mode": "ortho",
    "cut_coords": [5, -26, 21],
    "draw_cross": False,
    "vmax": 8,
    "cmap": "cold_hot",
}

# %%
# Comparing transparent and opaque thresholding
# ---------------------------------------------
#
# Here we use the motor activation image itself to give us the values
# to use for transparency.
# We can set ``transparency_range`` to ``[0.5, 3]``
# to range of values where transparency will be 'enabled'.
# Values below 0.5 will be fully transparent
# while values above 3 will be fully opaque.
#

plotting_config = {
    "display_mode": "ortho",
    "cut_coords": [5, -26, 21],
    "draw_cross": False,
    "vmax": 8,
    "cmap": "cold_hot",
}

fig, axes = plt.subplots(
    4,
    1,
    figsize=(figure_width, 17),
)

plot_stat_map(
    image,
    title="image without threshold",
    axes=axes[0],
    **plotting_config,
)
plot_stat_map(
    image,
    title="opaque thresholding",
    threshold=threshold,
    axes=axes[1],
    **plotting_config,
)
plot_stat_map(
    image,
    title="transparent thresholding",
    transparency=image,
    axes=axes[2],
    **plotting_config,
)
plot_stat_map(
    image,
    title="transparent thresholding with range",
    transparency=image,
    transparency_range=[vmin, threshold],
    axes=axes[3],
    **plotting_config,
)

show()

# %%
# Transparent thresholding and contours
# -------------------------------------
#
# If you want to visualize the limit where the transparency starts,
# you can add contours at the right threshold
# by using
# the :meth:`~nilearn.plotting.displays.BaseSlicer.add_contours` method.
#

fig, axes = plt.subplots(figsize=(figure_width, 4))

display = plot_stat_map(
    image,
    title="transparent thresholding with contour",
    transparency=image,
    transparency_range=[vmin, threshold],
    axes=axes,
    **plotting_config,
)
display.add_contours(
    image, filled=False, levels=[-threshold, threshold], colors=["k", "k"]
)

show()

# %%
# Transparent masking of part of the data
# ---------------------------------------
#
# You may want to use transparent masking to highlight
# specific parts of the brain while leaving other parts partly visible.
#
# For example, you could highlight the gray matter
# and leave values in the rest of the brain partly transparent.
#

# %%
# Let's fetch a beta image of auditory localizer from a single subject.
#

from nilearn.datasets import fetch_localizer_contrasts

auditory_image = fetch_localizer_contrasts(
    contrasts=["left auditory click"], verbose=0, n_subjects=1
)
auditory_image = auditory_image.cmaps[0]

# %%
# The let's create our transparency image
# to leave gray matter opaque and make the white matter
# partly transparent.
#
import numpy as np
from nibabel import Nifti1Image

from nilearn.datasets import load_mni152_gm_mask, load_mni152_wm_mask

white_matter_image = load_mni152_wm_mask(threshold=0.35)
white_matter_mask = white_matter_image.get_fdata() > 0

grey_matter_image = load_mni152_gm_mask(threshold=0.6)
grey_matter_mask = grey_matter_image.get_fdata() > 0

transparency_data = np.zeros(grey_matter_image.shape)
transparency_data[white_matter_mask] = 0.6
transparency_data[grey_matter_mask] = 1
transparency_image = Nifti1Image(transparency_data, grey_matter_image.affine)

# %%
# Create the plot.
#

fig, axes = plt.subplots(
    2,
    1,
    figsize=(figure_width, 8),
)

plotting_config = {
    "display_mode": "ortho",
    "cut_coords": [5, -26, 21],
    "draw_cross": False,
    "cmap": "cold_hot",
}

display = plot_stat_map(
    auditory_image,
    title="auditory localizer - no thresholding",
    axes=axes[0],
    **plotting_config,
)

display = plot_stat_map(
    auditory_image,
    title="auditory localizer  - highlight gray matter",
    transparency=transparency_image,
    axes=axes[1],
    **plotting_config,
)

show()

# %%
# .. note::
#
#    The transparency image was automatically resampled to the underlying data.
#

# %%
# Transparent thresholding with other functions
# ---------------------------------------------
#
# Several plotting functions support transparency including
# :func:`~nilearn.plotting.plot_glass_brain`,
# :func:`~nilearn.plotting.plot_stat_map` and
# :func:`~nilearn.plotting.plot_img`.
#
# See below an example with ``plot_glass_brain``.
#

from nilearn.plotting import plot_glass_brain

plotting_config = {
    "colorbar": True,
    "cmap": "inferno",
}

fig, axes = plt.subplots(
    4,
    1,
    figsize=(figure_width, 17),
)
plot_glass_brain(
    image,
    title="image without threshold",
    axes=axes[0],
    **plotting_config,
)
plot_glass_brain(
    image,
    title="opaque thresholding",
    threshold=threshold,
    axes=axes[1],
    **plotting_config,
)
plot_glass_brain(
    image,
    title="transparent thresholding",
    transparency=image,
    axes=axes[2],
    **plotting_config,
)
plot_glass_brain(
    image,
    title="transparent thresholding with range",
    transparency=image,
    transparency_range=[vmin, threshold],
    axes=axes[3],
    **plotting_config,
)

show()


# %%
# Transparent thresholding on GLM results
# ---------------------------------------
#
# You can also use different images as 'transparency' layer.
#
# For example, on the output of a GLM,
# you can visualize the contrast values and use their z-score as transparency.
#
# We will show this on a simple block paradigm GLM.
#
# .. seealso::
#
#     For more information
#     see the :ref:`dataset description <spm_auditory_dataset>`.
#
# In the following section we :
#
# - download the data,
#
# - fit the GLM with some smoothing of the data,
#
# - compute the contrast for the only condition present in this dataset,
#
# - compute the mean image of the functional data,
#   to use as underlay for our plots.
#
from nilearn.datasets import fetch_spm_auditory
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, show

subject_data = fetch_spm_auditory(verbose=0)

fmri_glm = FirstLevelModel(
    t_r=7,
    smoothing_fwhm=4,
    noise_model="ar1",
    standardize=False,
    hrf_model="spm",
    drift_model="cosine",
    high_pass=0.01,
)

fmri_glm = fmri_glm.fit(subject_data.func, subject_data.events)

results = fmri_glm.compute_contrast("listening", output_type="all")

mean_img = mean_img(subject_data.func[0], copy_header=True)

# %%
# Let's set some common configuration for our plots.
#
# We will look at activations only
# so we set ``vmin`` to 0 and use a sequential colormap (``inferno``).
plotting_config = {
    "bg_img": mean_img,
    "display_mode": "z",
    "cut_coords": [9, 42, 75],
    "black_bg": True,
    "vmin": 0,
    "cmap": "inferno",
}

# %%
# Here we will:
#
# - have a look at the statistical value for our contrast,
#
# - have a look at their Z score with opaque contrast,
#
# - use the Z score as transparency value,
#
# - finally we will threshold the Z-score to identify the significant clusters
#   (fdr=0.05, 500 voxels)
#   and plot those as contours.
#
fig, axes = plt.subplots(
    4,
    1,
    figsize=(figure_width, 18),
)

plot_stat_map(
    results["stat"],
    title="contrast value",
    axes=axes[0],
    **plotting_config,
)
plot_stat_map(
    results["z_score"],
    title="z-score, opaque threshold",
    threshold=3,
    axes=axes[1],
    **plotting_config,
)
plot_stat_map(
    results["stat"],
    title="contrast value, z-score as transparency",
    axes=axes[2],
    transparency=results["z_score"],
    **plotting_config,
)
display = plot_stat_map(
    results["stat"],
    title="contrast value, z-score as transparency, contoured clusters",
    axes=axes[3],
    transparency=results["z_score"],
    **plotting_config,
)
clean_map, threshold = threshold_stats_img(
    results["z_score"],
    alpha=0.05,
    height_control="fdr",
    cluster_threshold=500,
    two_sided=False,
)
display.add_contours(clean_map, filled=False, levels=[threshold], colors=["w"])

show()


# %%
# References
# ----------
#
# .. footbibliography::
#

# sphinx_gallery_dummy_images=2
# sphinx_gallery_thumbnail_number = 2
