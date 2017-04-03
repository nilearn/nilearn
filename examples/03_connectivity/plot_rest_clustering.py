"""
Clustering methods to learn a brain parcellation from rest fMRI
====================================================================

We use spatially-constrained Ward-clustering and KMeans to create a set
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
# We download one subject of the ADHD dataset from Internet

from nilearn import datasets
dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data


#########################################################################
# Nifti images to brain parcellations: Perform Ward Clustering
# -----------------------------------------------------------------

# Transforming list of images to data matrix takes few steps: reducing the
# data dimensionality using randomized SVD and build brain parcellations
# using specified clustering method KMeans or spatially-constrained Ward
# using a class named as `Parcellations`.

# import class Parcellations from nilearn.parcellations module
from nilearn.parcellations import Parcellations

# We build parameters of our own for this object. Parameters related to
# masking, caching and defining number of clusters and specific parcellations
# method.

# Computing ward for the first time, will be long... This can be seen by
# measuring using time
import time
start = time.time()

# Agglomerative Clustering: ward

# This object uses spatially-constrained AgglomerativeClustering for
# method='ward'. Spatial connectivity matrix (voxel-to-voxel) is built-in
# with this class which means no need of explicitly computing them.
ward = Parcellations(method='ward', n_parcels=1000,
                     standardize=False,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
# Call fit on functional dataset: single subject (less samples).
ward.fit(dataset.func)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute ward for second time to see the power of joblib caching with more
# parcels than first time

# This class also has built-in caching mechanism. Here, we compute ward
# clustering with more number of clusters=2000 and compare time with first
# time 1000 clusters.

# We initialize class again with n_parcels=2000 this time.
start = time.time()
ward = Parcellations(method='ward', n_parcels=2000,
                     standardize=False,
                     memory='nilearn_cache', memory_level=1,
                     verbose=1)
ward.fit(dataset.func)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

###########################################################################
# Visualize: Brain parcellations (Ward) and the effect of clustering
# -------------------------------------------------------------------
#
# First, we display the labels of the clustering in the brain.
#
# To visualize results, we need to transform the clustering's labels back
# to a neuroimaging volume space. For this, we use the masker's
# inverse_transform inherited from the class on output attribute `labels_`.
ward_labels_img = ward.masker_.inverse_transform(ward.labels_)

# Now, ward_labels_img are Nifti1Image object, it can be saved to file
# with the following code:
ward_labels_img.to_filename('ward_parcellation.nii')

from nilearn import plotting
from nilearn.image import mean_img, index_img

# we take mean over time on the functional image to use mean image as
# background to parcellated image assigned to ward_labels_img
mean_func_img = mean_img(dataset.func[0])

first_plot = plotting.plot_roi(ward_labels_img, mean_func_img,
                               title="Ward parcellation",
                               display_mode='xz')

# common cut coordinates for all plots
cut_coords = first_plot.cut_coords

# Second, we illustrate the effect that the clustering has on the signal.
# We show the original data, and the approximation provided by the
# clustering by averaging the signal on each parcel.

# Display the original data. We can use MultiNiftiMasker inherited from
# Parcellations object to transform Nifti file/files in a list to
# list of data matrix/matrices.
fmri_masked = ward.masker_.transform(dataset.func)
# grab number of voxels from fmri data matrix
original_voxels = fmri_masked[0].shape[1]

# common vmin and vmax
vmin = fmri_masked[0].min()
vmax = fmri_masked[0].max()

plotting.plot_epi(ward.masker_.inverse_transform(fmri_masked[0][0]),
                  cut_coords=cut_coords,
                  title='Original (%i voxels)' % original_voxels,
                  vmax=vmax, vmin=vmin, display_mode='xz')

# A reduced data can be create by taking the parcel-level average:
# Note that, the Parcellations object with any method has opportunity to
# use a transform method that modifies input features. Here it reduces their
# dimension. Note that we fit before calling a transform so that average signals
# can be created on the brain parcellations with fit call.
fmri_reduced = ward.transform(dataset.func)

# Display the corresponding data compressed using the parcellation using
# parcels=2000.
fmri_compressed = ward.inverse_transform(fmri_reduced)

plotting.plot_epi(index_img(fmri_compressed[0], 0),
                  cut_coords=cut_coords,
                  title='Ward compressed representation (2000 parcels)',
                  vmin=vmin, vmax=vmax, display_mode='xz')
# As you can see below, this approximation is almost good, although there
# are only 2000 parcels, instead of the original 60000 voxels

#########################################################################
# Now, Nifti images to brain parcellations: Perform KMeans Clustering
# -------------------------------------------------------------------
#
# We use the same approach as demonstrated with building parcellations using
# Ward clustering.

# class/functions can be used here as they are already imported above.

# We took parameters same as above for Parcellations object.

# Computing the kmeans and measure time duration for parcellations
import time
start = time.time()

# This object uses MiniBatchKMeans for method='kmeans'
kmeans = Parcellations(method='kmeans', n_parcels=2000,
                       standardize=False,
                       memory='nilearn_cache', memory_level=1,
                       verbose=1)
# Call fit on functional dataset: single subject (less samples)
kmeans.fit(dataset.func)
print("KMeans 2000 clusters: %.2fs" % (time.time() - start))
# NOTE: Good parcellations can be build using KMeans with more subjects,
# for instance more than 5 subjects

###########################################################################
# Visualize results Brain parcellations (KMeans) and effect of clustering
# -----------------------------------------------------------------------
#
# First, we display the labels of the clustering in the brain.
#
# To visualize results, same steps can be applied here following from Ward
# visualization.
kmeans_labels_img = kmeans.masker_.inverse_transform(kmeans.labels_)

plotting.plot_roi(kmeans_labels_img, mean_func_img,
                  title="KMeans parcellation",
                  display_mode='xz')

# kmeans_labels_img is a Nifti1Image object, it can be saved to file with
# the following code:
kmeans_labels_img.to_filename('kmeans_parcellation.nii')

##################################################################
# Second, we see the effect that the clustering has on the signal
#
# A reduced data can be create by taking the parcel-level average:
# We transform method that modifies input features. Here it reduces their
# dimension
fmri_reduced_kmeans = kmeans.transform(dataset.func)

# Display the corresponding data compressed using the kmeans parcellation
# with parcels=2000
fmri_compressed_kmeans = kmeans.inverse_transform(fmri_reduced_kmeans)

plotting.plot_epi(index_img(fmri_compressed_kmeans[0], 0),
                  cut_coords=cut_coords,
                  title='KMeans compressed representation (2000 parcels)',
                  vmin=vmin, vmax=vmax, display_mode='xz')

plotting.show()
