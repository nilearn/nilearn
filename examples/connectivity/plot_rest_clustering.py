"""
Ward clustering to learn a brain parcellation from rest fMRI
============================================================

We use spatially-constrained Ward-clustering to create a set of
parcels. These parcels are particularly interesting for creating a
'compressed' representation of the data, replacing the voxel-wise fMRI data
from nifti images by means of the derived gray-matter parcellation.

This parcellation may be useful for subsequent supervised learning.
See for instance:
`A supervised clustering approach for fMRI-based inference of
brain states <https://hal.inria.fr/inria-00589201>`_, Michel et al,
Pattern Recognition 2011.
"""

### Load nyu_rest dataset #####################################################

import numpy as np
from nilearn import datasets
from nilearn import input_data
from nilearn.plotting.img_plotting import plot_roi, plot_epi

nyu_dataset = datasets.fetch_nyu_rest(n_subjects=1)  # only first subject
# This is resting-state data: the background has not been removed yet.
# We therefore need to use mask_strategy='epi' to compute the mask from the
# EPI images
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache',
                                      mask_strategy='epi', memory_level=1,
                                      standardize=False)
func_path = nyu_dataset.func[0]
fmri_masked = nifti_masker.fit_transform(func_path)
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

### Ward ######################################################################

# Compute connectivity matrix: retrieve topology of voxel neighborhoods
from sklearn.feature_extraction import image
shape = mask.shape
connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)

# Computing the ward for the first time, this may take some time...
from sklearn.cluster import WardAgglomeration
import time
start = time.time()
ward = WardAgglomeration(n_clusters=1000, connectivity=connectivity,
                         memory='nilearn_cache', compute_full_tree=True)
ward.fit(fmri_masked)
print("Ward agglomeration 1000 clusters: %.2fs" % (time.time() - start))

# Compute the ward with more clusters, should be faster as we are using
# the caching mechanism (i.e., hierarchical tree is pre-computed)
start = time.time()
ward = WardAgglomeration(n_clusters=2000, connectivity=connectivity,
                         memory='nilearn_cache', compute_full_tree=True)
ward.fit(fmri_masked)
print("Ward agglomeration 2000 clusters: %.2fs" % (time.time() - start))

### Show result ###############################################################

# Unmask data (i.e., project the labels back into original space)
# Avoid 0 values to indicate a particular label
labels = ward.labels_ + 1
labels_img = nifti_masker.inverse_transform(labels)

# labels_img is a Nifti1Image object, it can be saved to disc by the
# following command:
labels_img.to_filename('parcellation.nii')

# plot the derived brain parcellation
from nilearn.image import mean_img
import matplotlib.pyplot as plt
mean_func_img = mean_img(func_path)

# common slices (i.e., cut coordinates) for all plots
first_plot = plot_roi(labels_img, mean_func_img, title="Ward parcellation",
                      display_mode='xz')

# Display the original data
plot_epi(nifti_masker.inverse_transform(fmri_masked[0]),
         cut_coords=first_plot.cut_coords,
         title='Original (%i voxels)' % fmri_masked.shape[1],
         display_mode='xz')

# MRI data compression can be performed by computing the parcel-wise average:
# Note that, as many objects in the scikit-learn, the ward object exposes
# a transform method that modifies input features. Here it reduces their
# dimension (i.e., individuals voxels) to parcels (i.e., groups of voxels)
fmri_reduced = ward.transform(fmri_masked)

# translate from label space to masked-brain space
fmri_compressed = ward.inverse_transform(fmri_reduced)
# translate from masked-brain space to original whole-brain space
compressed_img = nifti_masker.inverse_transform(fmri_compressed[0])

plot_epi(compressed_img, cut_coords=first_plot.cut_coords,
         title='Compressed representation (2000 parcels)',
         display_mode='xz')

plt.show()
