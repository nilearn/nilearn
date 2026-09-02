"""
Clustering methods to learn a brain parcellation from fMRI
==========================================================

We use spatially-constrained Ward-clustering, KMeans, Hierarchical KMeans
and Recursive Neighbor Agglomeration (ReNA) to create a set of parcels.

In a high dimensional regime, these methods can be interesting
to create a 'compressed' representation of the data,
replacing the data in the :term:`fMRI` images
by mean signals on the parcellation,
which can subsequently be used for statistical analysis or machine learning.

Also, these methods can be used to learn functional connectomes
and subsequently for classification tasks or to analyze data at a local level.

.. seealso::

    Which clustering method to use, an empirical comparison can be found
    in :footcite:t:`Thirion2014`.

    This :term:`parcellation` may be useful in a supervised learning,
    see for instance :footcite:t:`Michel2011b`.

    The big picture discussion corresponding to this example can be found
    in the documentation section :ref:`parcellating_brain`.
"""

# %%
# Download a brain development fMRI dataset
# -----------------------------------------
#
# We download one subject of the movie watching dataset.

import numpy as np

from nilearn import datasets

dataset = datasets.fetch_development_fmri(n_subjects=1)

# print basic information on the dataset
print(f"First subject functional nifti image (4D) is at: {dataset.func[0]}")


# %%
# Brain parcellation with Ward Clustering
# ---------------------------------------
#
# Transforming list of images to data matrix and build brain parcellation,
# all can be done at once using ``Parcellation`` objects.
#
# .. note::
#
#   Computing ward for the first time, will be long...
#   This can be seen by measuring using time
#
# We build parameters of our own for this object
# with parameters related to masking,
# caching and defining number of clusters and specific parcellation method.
#
import time

from nilearn.regions import Parcellations

start = time.time()
ward = Parcellations(
    method="ward",
    n_parcels=1000,
    smoothing_fwhm=2.0,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (fewer samples).
ward.fit(dataset.func)

# %%
print(f"Ward agglomeration 1000 clusters: {time.time() - start:.2f}s")

# %%
# We compute now ward clustering with 2000 clusters
# and compare time with 1000 clusters.
# To see the benefits of caching for second time.
#
# We initialize class again with ``n_parcels=2000`` this time.
#
start = time.time()
ward = Parcellations(
    method="ward",
    n_parcels=2000,
    smoothing_fwhm=2.0,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
ward.fit(dataset.func)

# %%
print(f"Ward agglomeration 2000 clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellation (Ward)
# ....................................
#
# First, we display the parcellation of the brain image
# stored in attribute ``labels_img_``.
from nilearn import plotting

ward_labels_img = ward.labels_img_

first_plot = plotting.plot_roi(
    ward_labels_img, title="Ward parcellation", display_mode="xz"
)

plotting.show()

# We grab the cut coordinates from this plot to use as a common for all plots.
cut_coords = first_plot.cut_coords

# %%
# Compressed representation of Ward clustering
# ............................................
#
# Second, we illustrate the effect that the clustering has on the signal.
# We show the original data,
# and the approximation provided by the clustering
# by averaging the signal on each parcel.

# Grab the number of voxels from attribute mask image (mask_img_).
from nilearn.image import get_data

original_voxels = np.sum(get_data(ward.mask_img_))

# Compute mean over time on the functional image to use the mean
# image for compressed representation comparisons
from nilearn.image import mean_img

mean_func_img = mean_img(dataset.func[0])

# Compute common vmin and vmax
vmin = np.min(get_data(mean_func_img))
vmax = np.max(get_data(mean_func_img))

plotting.plot_epi(
    mean_func_img,
    cut_coords=cut_coords,
    title=f"Original ({int(original_voxels)} voxels)",
    vmax=vmax,
    vmin=vmin,
    display_mode="xz",
)

# %
# A reduced dataset can be created by taking the parcel-level average.
#
# Parcellation objects with any method
# have the opportunity to use a ``transform`` call
# that modifies input features.
# Here it reduces their dimension.
# Note that we ``fit`` before calling a ``transform``
# so that average signals can be created on the brain parcellation
# with ``fit``.
#
fmri_reduced = ward.transform(dataset.func)

# Display the corresponding data compressed
# using the previous parcellation.
from nilearn.image import index_img

fmri_compressed = ward.inverse_transform(fmri_reduced)

plotting.plot_epi(
    index_img(fmri_compressed, 0),
    cut_coords=cut_coords,
    title=f"Ward compressed representation ({ward.n_parcels} parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)

plotting.show()

# %%
# As you can, this approximation is almost good,
# although there are only 2000 parcels, instead of the original 60000 voxels.
#

# %%
# Brain parcellation with KMeans Clustering
# -----------------------------------------
#
# We use the same approach as with building parcellation
# using Ward clustering.
# But, in the range of a small number of clusters,
# it is most likely that we want to use standardization.
# Indeed with standardization and smoothing, the clusters will form as regions.
#
# This next parcellation uses ``method='kmeans'`` for KMeans clustering
# with 10mm smoothing and standardization.
start = time.time()
kmeans = Parcellations(
    method="kmeans",
    n_parcels=50,
    smoothing_fwhm=10.0,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (fewer samples).
kmeans.fit(dataset.func)

# %%
print(f"KMeans clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellation (KMeans)
# ......................................
#
# We display the parcellation of the brain image
# stored in attribute ``labels_img_``.
kmeans_labels_img = kmeans.labels_img_

display = plotting.plot_roi(
    kmeans_labels_img,
    mean_func_img,
    cut_coords=cut_coords,
    title="KMeans parcellation",
    display_mode="xz",
)

plotting.show()

# %%
# Brain parcellation with Hierarchical KMeans Clustering
# ------------------------------------------------------
#
# As the number of images from which we try to cluster grows,
# voxels display more and more specific activity patterns
# causing KMeans clusters to be very unbalanced
# with a few big clusters and many voxels left as singletons.
#
# Hierarchical Kmeans algorithm is tailored
# to enforce more balanced clusterings.
# To do this,
# Hierarchical Kmeans does a first Kmeans clustering
# in square root of ``n_parcels``.
# In a second step, it clusters voxels inside each of these parcels
# in ``m`` pieces with ``m`` adapted to the size of the cluster
# in order to have n balanced clusters in the end.
#
# This object uses ``method='hierarchical_kmeans'``
# for Hierarchical KMeans clustering
# and 10mm smoothing and standardization to compare
# with the previous method.
start = time.time()
hkmeans = Parcellations(
    method="hierarchical_kmeans",
    n_parcels=50,
    smoothing_fwhm=10,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (fewer samples).
hkmeans.fit(dataset.func)

# %%
# Visualize: Brain parcellation (Hierarchical KMeans)
# ...................................................
#
# We display the parcellation of brain image
# stored in attribute ``labels_img_``.
hkmeans_labels_img = hkmeans.labels_img_

plotting.plot_roi(
    hkmeans_labels_img,
    mean_func_img,
    title="Hierarchical KMeans parcellation",
    display_mode="xz",
    cut_coords=display.cut_coords,
)

plotting.show()

# %%
# Compare Hierarchical Kmeans clusters with those from Kmeans
# ...........................................................
# To compare those, we'll first count how many voxels are contained
# in each of the 50 clusters for both algorithms
# and compare those sizes distribution.
# Hierarchical KMeans should give clusters
# closer to average (600 here) than KMeans.
#

# First count how many voxels have each label
# (except 0 which is the background).
_, kmeans_counts = np.unique(get_data(kmeans_labels_img), return_counts=True)

_, hkmeans_counts = np.unique(get_data(hkmeans_labels_img), return_counts=True)

voxel_ratio = np.round(np.sum(kmeans_counts[1:]) / 50)

# %%
# If all voxels not in background were balanced between clusters ...
print(f"... each cluster should contain {voxel_ratio} voxels")

# %%
# Let's plot clusters sizes distributions for both algorithms
#
# You can just skip the plotting code, the important part is the figure.
import matplotlib.pyplot as plt
from matplotlib import patches, ticker

bins = np.concatenate(
    [
        np.linspace(0, 500, 11),
        np.linspace(600, 2000, 15),
        np.linspace(3000, 10000, 8),
    ]
)
fig, axes = plt.subplots(
    nrows=2, sharex=True, gridspec_kw={"height_ratios": [4, 1]}
)
plt.semilogx()
axes[0].hist(kmeans_counts[1:], bins, color="blue")
axes[1].hist(hkmeans_counts[1:], bins, color="green")
axes[0].set_ylim(0, 16)
axes[1].set_ylim(4, 0)
axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
axes[1].yaxis.set_label_coords(-0.08, 2)
fig.subplots_adjust(hspace=0)
plt.xlabel("Number of voxels (log)", fontsize=12)
plt.ylabel("Number of clusters", fontsize=12)
handles = [
    patches.Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ["blue", "green"]
]
labels = ["Kmeans", "Hierarchical Kmeans"]
fig.legend(handles, labels, loc=(0.5, 0.8))

plotting.show()

# %%
# As we can see, half of the 50 KMeans clusters contain
# less than 100 voxels whereas three contain several thousands voxels.
# Hierarchical KMeans yield better balanced clusters,
# with a significant proportion of them containing hundreds
# to thousands of voxels.
#

# %%
# Brain parcellation with :term:`ReNA` Clustering
# -----------------------------------------------
#
# One interesting algorithmic property of :term:`ReNA` (see References)
# is that it is very fast
# for a large number of parcels (notably faster than Ward).
# As before, the :term:`parcellation` is done with a ``Parcellations`` object.
# The spatial constraints are implemented inside the ``Parcellations`` object.
#
# More about :term:`ReNA` clustering algorithm
# in the original paper (:footcite:t:`Hoyos2019`).
#
start = time.time()
rena = Parcellations(
    method="rena",
    n_parcels=5000,
    smoothing_fwhm=2.0,
    scaling=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

rena.fit_transform(dataset.func)

# %%
print(f"ReNA 5000 clusters: {time.time() - start:.2f}s")

# %%
# Visualize: Brain parcellation (ReNA)
# ....................................
#
# We display the parcellation of the brain image stored in attribute
# ``labels_img_``.
rena_labels_img = rena.labels_img_

plotting.plot_roi(
    ward_labels_img,
    title="ReNA parcellation",
    display_mode="xz",
    cut_coords=cut_coords,
)

plotting.show()

# %%
# Compressed representation of :term:`ReNA` clustering
# ....................................................
#
# We illustrate the effect that the clustering has on the signal.
# We show the original data, and the approximation provided by
# the clustering by averaging the signal on each parcel.
#
# We can then compare the results with the compressed representation
# obtained with Ward.

# Display the original data
plotting.plot_epi(
    mean_func_img,
    cut_coords=cut_coords,
    title=f"Original ({int(original_voxels)} voxels)",
    vmax=vmax,
    vmin=vmin,
    display_mode="xz",
)

# A reduced data can be created by taking the parcel-level average:
# Note that, as many scikit-learn objects, the ``rena`` object exposes
# a ``transform`` method that modifies input features.
# Here it reduces their dimension.
# However, the data are in one single large 4D image, we need to use
# :func:`~nilearn.image.index_img` to do the split easily:
fmri_reduced_rena = rena.transform(dataset.func)

# Display the corresponding data compression using the parcellation
compressed_img_rena = rena.inverse_transform(fmri_reduced_rena)

plotting.plot_epi(
    index_img(compressed_img_rena, 0),
    cut_coords=cut_coords,
    title=f"ReNA compressed representation ({rena.n_parcels} parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)

plotting.show()

# %%
# Even if the compressed signal is relatively close to the original signal,
# we can notice that Ward Clustering
# gives a slightly more accurate compressed representation.
# However, as said in the previous section,
# the computation time is reduced
# which could still make :term:`ReNA` more relevant than Ward in some cases.

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=3
