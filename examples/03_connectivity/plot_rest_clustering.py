"""
Ward clustering to learn a brain parcellation from rest fMRI
====================================================================

We use spatially-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for creating a
'compressed' representation of the data, replacing the data in the fMRI
images by mean on the parcellation.

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


##################################################################
# Transform nifti files to a data matrix with the NiftiMasker
from nilearn import input_data

# The NiftiMasker will extract the data on a mask. We do not have a
# mask, hence we need to compute one.
#
# This is resting-state data: the background has not been removed yet,
# thus we need to use mask_strategy='epi' to compute the mask from the
# EPI images
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)

func_filename = dataset.func[0]
# The fit_transform call computes the mask and extracts the time-series
# from the files:
fmri_masked = nifti_masker.fit_transform(func_filename)

# We can retrieve the numpy array of the mask
mask = nifti_masker.mask_img_.get_data().astype(bool)


##################################################################
# Perform Ward clustering
# -----------------------
#
# We use spatially-constrained Ward clustering. For this, we need to
# compute from the mask a matrix giving the voxel-to-voxel connectivity

# Compute connectivity matrix: which voxel is connected to which
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)


##################################################################
# Then we use FeatureAgglomeration from scikit-learn. Indeed, the voxels
# are the features of the data matrix.
#
# In addition, we use caching. As a result, the clustering doesn't have
# to be recomputed later.

# Computing the ward for the first time, this is long...
from sklearn.cluster import FeatureAgglomeration
# If you have scikit-learn older than 0.14, you need to import
# WardAgglomeration instead of FeatureAgglomeration
import time
start = time.time()
ward = FeatureAgglomeration(n_clusters=1000, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(fmri_masked)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism
start = time.time()
ward = FeatureAgglomeration(n_clusters=2000, connectivity=connectivity,
                            linkage='ward', memory='nilearn_cache')
ward.fit(fmri_masked)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

##################################################################
# Visualize results
# ------------------
#
# First we display the labels of the clustering in the brain.
#
# To visualize results, we need to transform the clustering's labels back
# to a neuroimaging volume. For this, we use the NiftiMasker's
# inverse_transform method.
from nilearn.plotting import plot_roi, plot_epi, show

# Unmask the labels

# Avoid 0 label
labels = ward.labels_ + 1
labels_img = nifti_masker.inverse_transform(labels)

from nilearn.image import mean_img
mean_func_img = mean_img(func_filename)


first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation",
                      display_mode='xz')

# common cut coordinates for all plots
cut_coords = first_plot.cut_coords

##################################################################
# labels_img is a Nifti1Image object, it can be saved to file with the
# following code:
labels_img.to_filename('parcellation.nii')


##################################################################
# Second, we illustrate the effect that the clustering has on the
# signal. We show the original data, and the approximation provided by
# the clustering by averaging the signal on each parcel.
#
# As you can see below, this approximation is very good, although there
# are only 2000 parcels, instead of the original 60000 voxels

# Display the original data
plot_epi(nifti_masker.inverse_transform(fmri_masked[0]),
         cut_coords=cut_coords,
         title='Original (%i voxels)' % fmri_masked.shape[1],
         vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')

# A reduced data can be create by taking the parcel-level average:
# Note that, as many objects in the scikit-learn, the ward object exposes
# a transform method that modifies input features. Here it reduces their
# dimension
fmri_reduced = ward.transform(fmri_masked)

# Display the corresponding data compressed using the parcellation
fmri_compressed = ward.inverse_transform(fmri_reduced)
compressed_img = nifti_masker.inverse_transform(fmri_compressed[0])

plot_epi(compressed_img, cut_coords=cut_coords,
         title='Compressed representation (2000 parcels)',
         vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')

show()
