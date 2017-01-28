"""
Clustering methods to learn a brain parcellation from rest fMRI
====================================================================

We use KMeans and spatially-constrained Ward-clustering to create a set
of parcels.

In a high dimensional regime, these methods are particularly interesting
for creating a 'compressed' representation of the data, replacing the data
in the fMRI images by mean on the parcellation.

On the other way, these methods will also be interesting for learning
functional connectomes based on these parcellations and be able to used
in a classification task between controls and disease states.

This parcellation may be useful in a supervised learning, see for
instance: `A supervised clustering approach for fMRI-based inference of
brain states <https://hal.inria.fr/inria-00589201>`_, Michel et al,
Pattern Recognition 2011.

The big picture discussion corresponding to this example can be found
in the documentation section :ref:`parcellating_brain`.
"""

##################################################################
# Download a rest dataset and turn it to a data matrix
# -----------------------------------------------------
#
# We we download one subject of the ADHD dataset from Internet

from nilearn import datasets
dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data


#########################################################################
# Nifti images to brain parcellations: Perform KMeans Clustering
# ---------------------------------------------------------------
#
# Transforming images to data matrix takes few steps: reducing the data
# dimensionality using randomized SVD and build brain parcellations using
# specified clustering method KMeans or spatially-constrained Ward using
# a class named as `Parcellations`.

# import parcellations from nilearn
from nilearn.parcellations import Parcellations

# We build parameters of our own for this object. Parameters related to
# masking, caching and defining number of clusters and specific method.

# Computing the kmeans and measure time duration for parcellations
import time
start = time.time()

# This object uses MiniBatchKMeans for method='kmeans'
kmeans = Parcellations(method='kmeans', n_parcels=1000,
                       standardize=False,
                       memory='nilearn_cache', memory_level=1,
                       verbose=1)
# Call fit on functional dataset: single subject (less samples)
kmeans.fit(dataset.func)
print("KMeans 1000 clusters: %.2fs" % (time.time() - start))
# NOTE: Good parcellations can be build using KMeans with more subjects,
# for instance more than 5 subjects

#########################################################################
# Now, Nifti images to brain parcellations: Perform Ward Clustering
# -----------------------------------------------------------------

# Agglomerative Clustering: ward

# Computing ward for the first time, will be long... This can be seen by
# measuring using time
start = time.time()

# This object uses spatially-constrained AgglomerativeClustering for
# method='ward'. Spatial connectivity matrix (voxel-to-voxel) is built-in
# with this class which means no need of explicitly compute them.
ward = Parcellations(method='ward', n_parcels=1000,
                     standardize=False,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
ward.fit(dataset.func)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute ward for second time to see the power of joblib caching

# This class also has built-in caching mechanism. Here, we compute ward
# clustering with more number of clusters=2000 and compare time with first
# time 1000 clusters.
start = time.time()
ward = Parcellations(method='ward', n_parcels=2000,
                     standardize=False, memory='nilearn_cache',
                     memory_level=1, verbose=1)
ward.fit(dataset.func)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

###########################################################################
# Visualize results
# ------------------
#
# First, we display the labels of the clustering in the brain.
#
# To visualize results, we need to transform the clustering's labels back
# to a neuroimaging volume. For this, we use the masker's inverse_transform
# directly from the object on attribute labels_.

kmeans_labels_img = kmeans.masker_.inverse_transform(kmeans.labels_)
ward_labels_img = ward.masker_.inverse_transform(ward.labels_)

from nilearn.plotting import plot_roi, plot_epi, show
from nilearn.image import mean_img

# we take mean over time on the functional image to use mean image as
# background to parcellated image labels_img
mean_func_img = mean_img(dataset.func[0])


first_plot = plot_roi(kmeans_labels_img, mean_func_img,
                      title="KMeans parcellation",
                      display_mode='xz')
second_plot = plot_roi(ward_labels_img, mean_func_img,
                       title="Ward parcellation",
                       display_mode='xz')

# common cut coordinates for all plots
cut_coords = first_plot.cut_coords

##################################################################
# kmeans_labels_img and ward_labels_img are Nifti1Image object, it can be
# saved to file with the following code:
kmeans_labels_img.to_filename('kmeans_parcellation.nii')
ward_labels_img.to_filename('ward_parcellation.nii')

##################################################################
# Second, we illustrate the effect that the clustering has on the
# signal. We show the original data, and the approximation provided by
# the clustering by averaging the signal on each parcel.
#
# As you can see below, this approximation is very good, although there
# are only 2000 parcels, instead of the original 60000 voxels

# Display the original data
fmri_masked = ward.masker_.transform(dataset.func)
plot_epi(ward.masker_.inverse_transform(fmri_masked[0][0]),
         cut_coords=cut_coords,
         title='Original (%i voxels)' % fmri_masked[0].shape[1],
         vmax=fmri_masked[0].max(), vmin=fmri_masked[0].min(),
         display_mode='xz')

# Display the corresponding data compressed using the parcellation using
# parcels=2000
plot_epi(ward_labels_img, cut_coords=cut_coords,
         title='Compressed representation (2000 parcels)',
         display_mode='xz')

show()
