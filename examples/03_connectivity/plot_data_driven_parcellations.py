"""
Clustering methods to learn a brain parcellation from fMRI
==========================================================

We use spatially-constrained Ward-clustering, KMeans, Hierarchical KMeans
and Recursive Neighbor Agglomeration (ReNA) to create a set of parcels.

In a high dimensional regime, these methods can be interesting
to create a 'compressed' representation of the data, replacing the data
in the fMRI images by mean signals on the parcellation, which can
subsequently be used for statistical analysis or machine learning.

Also, these methods can be used to learn functional connectomes
and subsequently for classification tasks or to analyze data at a local
level.

References
----------
Which clustering method to use, an empirical comparison can be found in this
paper:

Bertrand Thirion, Gael Varoquaux, Elvis Dohmatob, Jean-Baptiste Poline.
  `Which fMRI clustering gives good brain parcellations ?
  <https://doi.org/10.3389/fnins.2014.00167>`_ Frontiers in Neuroscience,
  2014.

This parcellation may be useful in a supervised learning, see for
instance:

Vincent Michel, Alexandre Gramfort, Gael Varoquaux, Evelyn Eger,
  Christine Keribin, Bertrand Thirion. `A supervised clustering approach
  for fMRI-based inference of brain states.
  <http://dx.doi.org/10.1016/j.patcog.2011.04.006>`_.
  Pattern Recognition, Elsevier, 2011.

The big picture discussion corresponding to this example can be found
in the documentation section :ref:`parcellating_brain`.
"""

########################################################################
# Download a brain development fmri dataset and turn it to a data matrix
# -----------------------------------------------------------------------
#
# We download one subject of the movie watching dataset from Internet

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, ticker

from nilearn import datasets, plotting
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations

dataset = datasets.fetch_development_fmri(n_subjects=1)

# print basic information on the dataset
print(
    f"First subject functional nifti image (4D) is at: {dataset.func[0]}"
)


#########################################################################
# Brain parcellations with Ward Clustering
# ----------------------------------------
#
# Transforming list of images to data matrix and build brain parcellations,
# all can be done at once using `Parcellations` object.


# Computing ward for the first time, will be long... This can be seen by
# measuring using time
start = time.time()

# Agglomerative Clustering: ward

# We build parameters of our own for this object. Parameters related to
# masking, caching and defining number of clusters and specific parcellations
# method.
ward = Parcellations(
    method="ward",
    n_parcels=1000,
    standardize=False,
    smoothing_fwhm=2.0,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (less samples).
ward.fit(dataset.func)
print(f"Ward agglomeration 1000 clusters: {time.time() - start:.2f}s")

# We compute now ward clustering with 2000 clusters and compare
# time with 1000 clusters. To see the benefits of caching for second time.

# We initialize class again with n_parcels=2000 this time.
start = time.time()
ward = Parcellations(
    method="ward",
    n_parcels=2000,
    standardize=False,
    smoothing_fwhm=2.0,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
ward.fit(dataset.func)
print(f"Ward agglomeration 2000 clusters: {time.time() - start:.2f}s")

###########################################################################
# Visualize: Brain parcellations (Ward)
# .....................................
#
# First, we display the parcellations of the brain image stored in attribute
# `labels_img_`
ward_labels_img = ward.labels_img_

# Now, ward_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
ward_labels_img.to_filename("ward_parcellation.nii.gz")


first_plot = plotting.plot_roi(
    ward_labels_img, title="Ward parcellation", display_mode="xz"
)

# Grab cut coordinates from this plot to use as a common for all plots
cut_coords = first_plot.cut_coords

###########################################################################
# Compressed representation of Ward clustering
# ............................................
#
# Second, we illustrate the effect that the clustering has on the signal.
# We show the original data, and the approximation provided by the
# clustering by averaging the signal on each parcel.

# Grab number of voxels from attribute mask image (mask_img_).
original_voxels = np.sum(get_data(ward.mask_img_))

# Compute mean over time on the functional image to use the mean
# image for compressed representation comparisons
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

# A reduced dataset can be created by taking the parcel-level average:
# Note that Parcellation objects with any method have the opportunity to
# use a `transform` call that modifies input features. Here it reduces their
# dimension. Note that we `fit` before calling a `transform` so that average
# signals can be created on the brain parcellations with fit call.
fmri_reduced = ward.transform(dataset.func)

# Display the corresponding data compressed using the parcellation using
# parcels=2000.
fmri_compressed = ward.inverse_transform(fmri_reduced)

plotting.plot_epi(
    index_img(fmri_compressed, 0),
    cut_coords=cut_coords,
    title="Ward compressed representation (2000 parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)
# As you can see below, this approximation is almost good, although there
# are only 2000 parcels, instead of the original 60000 voxels

#########################################################################
# Brain parcellations with KMeans Clustering
# ------------------------------------------
#
# We use the same approach as with building parcellations using Ward
# clustering. But, in the range of a small number of clusters,
# it is most likely that we want to use standardization. Indeed with
# standardization and smoothing, the clusters will form as regions.

# class/functions can be used here as they are already imported above.

# This object uses method='kmeans' for KMeans clustering with 10mm smoothing
# and standardization ON
start = time.time()
kmeans = Parcellations(
    method="kmeans",
    n_parcels=50,
    standardize=True,
    smoothing_fwhm=10.0,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (less samples)
kmeans.fit(dataset.func)
print(f"KMeans clusters: {time.time() - start:.2f}s")

###########################################################################
# Visualize: Brain parcellations (KMeans)
# .......................................
#
# Grab parcellations of brain image stored in attribute `labels_img_`
kmeans_labels_img = kmeans.labels_img_

display = plotting.plot_roi(
    kmeans_labels_img,
    mean_func_img,
    title="KMeans parcellation",
    display_mode="xz",
)

# kmeans_labels_img is a Nifti1Image object, it can be saved to file with
# the following code:
kmeans_labels_img.to_filename("kmeans_parcellation.nii.gz")

#########################################################################
# Brain parcellations with Hierarchical KMeans Clustering
# -------------------------------------------------------
#
# As the number of images from which we try to cluster grows,
# voxels display more and more specific activity patterns causing
# KMeans clusters to be very unbalanced with a few big clusters and
# many voxels left as singletons. Hierarchical Kmeans algorithm is
# tailored to enforce more balanced clusterings. To do this,
# Hierarchical Kmeans does a first Kmeans clustering in square root of
# n_parcels. In a second step, it clusters voxels inside each
# of these parcels in m pieces with m adapted to the size of
# the cluster in order to have n balanced clusters in the end.
#
# This object uses method='hierarchical_kmeans' for Hierarchical KMeans
# clustering and 10mm smoothing and standardization to compare
# with the previous method.
start = time.time()
hkmeans = Parcellations(
    method="hierarchical_kmeans",
    n_parcels=50,
    standardize=True,
    smoothing_fwhm=10,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
# Call fit on functional dataset: single subject (less samples)
hkmeans.fit(dataset.func)

###########################################################################
# Visualize: Brain parcellations (Hierarchical KMeans)
# ....................................................
#
# Grab parcellations of brain image stored in attribute `labels_img_`
hkmeans_labels_img = hkmeans.labels_img_

plotting.plot_roi(
    hkmeans_labels_img,
    mean_func_img,
    title="Hierarchical KMeans parcellation",
    display_mode="xz",
    cut_coords=display.cut_coords,
)

# kmeans_labels_img is a :class:`nibabel.nifti1.Nifti1Image` object, it can be
# saved to file with the following code:
hkmeans_labels_img.to_filename("hierarchical_kmeans_parcellation.nii.gz")

###########################################################################
# Compare Hierarchical Kmeans clusters with those from Kmeans
# ...........................................................
# To compare those, we'll first count how many voxels are contained in
# each of the 50 clusters for both algorithms and compare those sizes
# distribution. Hierarchical KMeans should give clusters closer to
# average (600 here) than KMeans.
#
# First count how many voxels have each label (except 0 which is the
# background).

_, kmeans_counts = np.unique(
    get_data(kmeans_labels_img), return_counts=True
)

_, hkmeans_counts = np.unique(get_data(hkmeans_labels_img), return_counts=True)

voxel_ratio = np.round(np.sum(kmeans_counts[1:]) / 50)

# If all voxels not in background were balanced between clusters ...

print(f"... each cluster should contain {voxel_ratio} voxels")

###########################################################################
# Let's plot clusters sizes distributions for both algorithms
#
# You can just skip the plotting code, the important part is the figure


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
###########################################################################
# As we can see, half of the 50 KMeans clusters contain less than
# 100 voxels whereas three contain several thousands voxels
# Hierarchical KMeans yield better balanced clusters, with a significant
# proportion of them containing hundreds to thousands of voxels.
#

###########################################################################
# Brain parcellations with ReNA Clustering
# ----------------------------------------
#
# One interesting algorithmic property of ReNA (see References) is that
# it is very fast for a large number of parcels (notably faster than Ward).
# As before, the parcellation is done with a Parcellations object.
# The spatial constraints are implemented inside the Parcellations object.
#
# References
# ..........
#
# More about ReNA clustering algorithm in the original paper
#
#     * A. Hoyos-Idrobo, G. Varoquaux, J. Kahn and B. Thirion, "Recursive
#       Nearest Agglomeration (ReNA): Fast Clustering for Approximation of
#       Structured Signals," in IEEE Transactions on Pattern Analysis and
#       Machine Intelligence, vol. 41, no. 3, pp. 669-681, 1 March 2019.
#       https://hal.archives-ouvertes.fr/hal-01366651/
start = time.time()
rena = Parcellations(
    method="rena",
    n_parcels=5000,
    standardize=False,
    smoothing_fwhm=2.0,
    scaling=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

rena.fit_transform(dataset.func)
print(f"ReNA 5000 clusters: {time.time() - start:.2f}s")

###########################################################################
# Visualize: Brain parcellations (ReNA)
# .....................................
#
# First, we display the parcellations of the brain image stored in attribute
# `labels_img_`
rena_labels_img = rena.labels_img_

# Now, rena_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
rena_labels_img.to_filename("rena_parcellation.nii.gz")

plotting.plot_roi(
    ward_labels_img,
    title="ReNA parcellation",
    display_mode="xz",
    cut_coords=cut_coords,
)

###########################################################################
# Compressed representation of ReNA clustering
# ............................................
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
# Note that, as many scikit-learn objects, the ReNA object exposes
# a transform method that modifies input features. Here it reduces their
# dimension.
# However, the data are in one single large 4D image, we need to use
# index_img to do the split easily:
fmri_reduced_rena = rena.transform(dataset.func)

# Display the corresponding data compression using the parcellation
compressed_img_rena = rena.inverse_transform(fmri_reduced_rena)

plotting.plot_epi(
    index_img(compressed_img_rena, 0),
    cut_coords=cut_coords,
    title="ReNA compressed representation (5000 parcels)",
    vmin=vmin,
    vmax=vmax,
    display_mode="xz",
)

###########################################################################
# Even if the compressed signal is relatively close
# to the original signal, we can notice that Ward Clustering
# gives a slightly more accurate compressed representation.
# However, as said in the previous section, the computation time is
# reduced which could still make ReNA more relevant than Ward in
# some cases.
