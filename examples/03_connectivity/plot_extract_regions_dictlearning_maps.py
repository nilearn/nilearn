"""
Regions extraction using dictionary learning and functional connectomes
=======================================================================

This example shows how to use :class:`~nilearn.regions.RegionExtractor`
to extract spatially constrained brain regions from whole brain maps decomposed
using :term:`Dictionary learning` and use them to build
a :term:`functional connectome`.

We used movie-watching functional scans of 20 subjects from
:func:`~nilearn.datasets.fetch_development_fmri` and
:class:`~nilearn.decomposition.DictLearning` for set of brain atlas maps.

This example can also be inspired to apply the same steps
to even regions extraction
using :term:`ICA` maps.
In that case, the idea would be to replace
:term:`Dictionary learning` to canonical :term:`ICA` decomposition
using :class:`~nilearn.decomposition.CanICA`

Please see the related documentation
of :class:`~nilearn.regions.RegionExtractor` for more details.

"""

# %%
# Fetch brain development functional datasets
# -------------------------------------------
#
# We use nilearn's datasets downloading utilities
from nilearn.datasets import fetch_development_fmri

rest_dataset = fetch_development_fmri(n_subjects=20)
func_filenames = rest_dataset.func
confounds = rest_dataset.confounds

# %%
# Extract functional networks with :term:`Dictionary learning`
# ------------------------------------------------------------
#
# Import :class:`~nilearn.decomposition.DictLearning` from the
# :mod:`~nilearn.decomposition` module, instantiate the object, and
# :meth:`~nilearn.decomposition.DictLearning.fit` the model to the
# functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(
    n_components=8,
    smoothing_fwhm=6.0,
    memory="nilearn_cache",
    memory_level=1,
    random_state=0,
    verbose=1,
)
# Fit to the data
dict_learn.fit(func_filenames)

# %%
# Visualization of functional networks
# ....................................
# Get the extracted functional networks
# via the attribute ``components_img_``
# and show them using plotting utilities.
#
from nilearn.plotting import plot_prob_atlas, show

components_img = dict_learn.components_img_

plot_prob_atlas(
    components_img,
    view_type="filled_contours",
    title="Dictionary Learning maps",
    draw_cross=False,
)

show()

# %%
# Extract regions from networks
# -----------------------------
#
# Import :class:`~nilearn.regions.RegionExtractor` from the
# :mod:`~nilearn.regions` module.
# ``threshold=0.5`` indicates that we keep nominal of amount nonzero
# :term:`voxels<voxel>` across all maps, less the threshold means that
# Import :class:`~nilearn.regions.RegionExtractor` from the
# :mod:`~nilearn.regions` module.
# ``threshold=0.5`` indicates that we keep a nominal of amount non-zero
# :term:`voxels<voxel>` across all maps.
#
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(
    components_img,
    threshold=0.5,
    standardize="zscore_sample",
    thresholding_strategy="ratio_n_voxels",
    extractor="local_regions",
    standardize_confounds=True,
    min_region_size=1350,
    verbose=1,
)

# Just call fit() to proceed with regions extraction.
extractor.fit()

# %%
# Visualization of region extraction results
# ..........................................
# Extracted regions are stored in ``regions_img_``
# and each region index is stored in ``index_``.
regions_index = extractor.index_

regions_extracted_img = extractor.regions_img_
n_regions_extracted = regions_extracted_img.shape[-1]
n_components = components_img.shape[-1]
title = (
    f"{n_regions_extracted} regions are extracted "
    f"from {n_components} components.\n"
    "Each separate color of region indicates extracted region."
)

plot_prob_atlas(
    regions_extracted_img,
    view_type="filled_contours",
    title=title,
    draw_cross=False,
)

show()

# %%
# Compute correlation coefficients
# --------------------------------
#
# First we need to do subjects timeseries signals extraction
# and then estimating correlation matrices on those signals.
# To extract timeseries signals, we call
# :meth:`~nilearn.regions.RegionExtractor.transform` onto each subject
# functional data stored in ``func_filenames``.
# First we need to extract timeseries signals for each subject
# and then estimate the correlation matrices on those signals.
# To extract timeseries, we call
# :meth:`~nilearn.regions.RegionExtractor.transform` onto each subject
# functional data stored in ``func_filenames``.
# To estimate the correlation matrices
# we can rely on :class:`nilearn.connectome.ConnectivityMeasure`.
from nilearn.connectome import ConnectivityMeasure

connectome_measure = ConnectivityMeasure(kind="correlation", verbose=1)

correlations = []
for filename, confound in zip(func_filenames, confounds, strict=False):
    timeseries_each_subject = extractor.transform(filename, confounds=confound)
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    correlations.append(correlation)


# %%
# Plot resulting connectomes
# --------------------------
#
# First we plot the mean of correlation matrices with
# :func:`~nilearn.plotting.plot_matrix`,
# and we use :func:`~nilearn.plotting.plot_connectome`
# to plot the connectome relations.
import numpy as np

from nilearn.plotting import (
    find_probabilistic_atlas_cut_coords,
    find_xyz_cut_coords,
    plot_connectome,
    plot_matrix,
)

mean_correlations = np.mean(correlations, axis=0).reshape(
    n_regions_extracted, n_regions_extracted
)

title = f"Correlation between {int(n_regions_extracted)} regions"
plot_matrix(mean_correlations, vmax=1, vmin=-1, title=title)

# Find the center of the regions and plot the connectome.
regions_img = regions_extracted_img
coords_connectome = find_probabilistic_atlas_cut_coords(regions_img)
plot_connectome(
    mean_correlations, coords_connectome, edge_threshold="90%", title=title
)

show()

# %%
# Plot regions extracted for only one specific network
# ----------------------------------------------------
#
# First, we plot a network of ``index=4``
# without region extraction.
from nilearn import image
from nilearn.plotting import plot_stat_map

index = 4

img = image.index_img(components_img, index)
coords = find_xyz_cut_coords(img)
plot_stat_map(
    img,
    cut_coords=coords,
    title="Showing one specific network",
)

show()

# %%
# Now, we plot the same network after region extraction to show that
# connected regions are nicely separated.
# Each brain extracted region is identified as separate color.
#
# For this, we take the indices of the all regions extracted
# related to original network given as 4.

from nilearn.plotting import cm, plot_anat

regions_indices_of_map3 = np.where(np.array(regions_index) == index)

display = plot_anat(
    cut_coords=coords, title="Regions from this network", colorbar=False
)

# Add as an overlay all the regions of index 4
colors = "rgbcmyk"
for each_index_of_map3, color in zip(
    regions_indices_of_map3[0], colors, strict=False
):
    display.add_overlay(
        image.index_img(regions_extracted_img, each_index_of_map3),
        cmap=cm.alpha_cmap(color),
    )

show()

# sphinx_gallery_dummy_images=6
