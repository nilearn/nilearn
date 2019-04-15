"""
ReNA clustering to learn a brain parcellation from rest fMRI
====================================================================

We use Recursive Neighbor Agglomeration (ReNA) clustering to create a set of
parcels. These parcels are particularly interesting for creating a 'compressed'
representation of the data, replacing the data in the fMRI images by mean on
the parcellation.

This parcellation may be useful in a supervised learning, see for
instance: `A supervised clustering approach for fMRI-based inference of
brain states <https://hal.inria.fr/inria-00589201>`_, Michel et al,
Pattern Recognition 2011.

The computation time of this algorithm is linear in the number of features.
This makes it well suited to use in the consensus on several random
parcellations, see for instance: `Randomized parcellation based inference
<https://hal.inria.fr/hal-00915243>`_, Da Mota et al, Neuroimage 2014.

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
# The fit call computes the mask
nifti_masker.fit(func_filename)

##################################################################
# Perform ReNA clustering
# -----------------------
# The spatial constraints are implemented inside the ReNA object.
#
# We use caching. As a result, the clustering doesn't have
# to be recomputed later.
import time
from nilearn.connectome import ReNA
start = time.time()
rena = ReNA(scaling=True, n_clusters=1000, mask=nifti_masker,
            memory='nilearn_cache')

rena.fit_transform(func_filename)
print("ReNA 1000 clusters: %.2fs" % (time.time() - start))

start = time.time()
rena = ReNA(scaling=True, n_clusters=2000, mask=nifti_masker,
            memory='nilearn_cache')

rena.fit_transform(func_filename)
print("ReNA 2000 clusters: %.2fs" % (time.time() - start))

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
import numpy as np

# Avoid 0 label
labels = rena.labels_ + 1
# Shuffling the labels for visualization
permutation = np.random.permutation(labels.shape[0])
labels = permutation[labels]
# Unmask the labels
labels_img = nifti_masker.inverse_transform(labels)

##################################################################
# labels_img is a Nifti1Image object, it can be saved to file with the
# following code:
labels_img.to_filename('parcellation.nii')


from nilearn.image import mean_img
mean_func_img = mean_img(func_filename)

first_plot = plot_roi(labels_img, mean_func_img, title="ReNA parcellation",
                      display_mode='xz')

# common cut coordinates for all plots
cut_coords = first_plot.cut_coords

##################################################################
# Second, we illustrate the effect that the clustering has on the
# signal. We show the original data, and the approximation provided by
# the clustering by averaging the signal on each parcel.
#
# As you can see below, this approximation is very good, although there
# are only 2000 parcels, instead of the original 60000 voxels

# The transform call extracts the time-series from the files:
fmri_masked = nifti_masker.transform(func_filename)

# Display the original data
plot_epi(nifti_masker.inverse_transform(fmri_masked[0]),
         title='Original (%i voxels)' % fmri_masked.shape[1],
         cut_coords=cut_coords, vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')


# A reduced data can be created by taking the parcel-level average:
# Note that, as many scikit-learn objects, the ReNA object exposes
# a transform method that modifies input features. Here it reduces their
# dimension.
# However, the data are in one single large 4D image, we need to use
# index_img to do the split easily:
from nilearn.image import index_img
fmri_reduced_rena = rena.transform(index_img(func_filename, 0))

# Display the corresponding data compression using the parcellation
compressed_img_rena = rena.inverse_transform(fmri_reduced_rena)

plot_epi(compressed_img_rena, cut_coords=cut_coords,
         title='ReNA: Compressed representation (2000 parcels)',
         vmax=fmri_masked.max(), vmin=fmri_masked.min(),
         display_mode='xz')

show()
