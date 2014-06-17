"""
Computing an ROI mask
=======================

Example showing how a T-test can be performed to compute an ROI
mask, and how simple operations can improve the quality of the mask
obtained.
"""
### Coordinates of the selected slice #########################################

coronal = -24
sagittal = -33
axial = -17

### Load the data #############################################################

# Fetch the data files from Internet
from nilearn import datasets
import nibabel
haxby_files = datasets.fetch_haxby(n_subjects=1)

# Second, load the labels
import numpy as np
labels = np.genfromtxt(haxby_files.session_target[0], skip_header=1,
                          usecols=[0], dtype=basestring)

### Visualization function ####################################################

import matplotlib.pyplot as plt
from nilearn.plotting import plot_epi, plot_stat_map, plot_roi


### Find voxels of interest ###################################################

# Smooth the data
from nilearn import image
fmri_img = image.smooth_img(haxby_files.func[0], fwhm=6)

# Plot the mean image
mean_img = image.mean_img(fmri_img)
plot_epi(mean_img, title='Smoothed mean EPI', cut_coords=(coronal, sagittal,
                                                          axial))

# Run a T-test for face and houses
from scipy import stats
fmri_data = fmri_img.get_data()
_, pvalues = stats.ttest_ind(fmri_data[..., labels == 'face'],
                             fmri_data[..., labels == 'house'], axis=-1)

# Use a log scale for p-values
pvalues = - np.log10(pvalues)
pvalues[np.isnan(pvalues)] = 0.
pvalues[pvalues > 10] = 10
plot_stat_map(nibabel.Nifti1Image(pvalues, fmri_img.get_affine()), mean_img,
              title="p-values", cut_coords=(coronal, sagittal, axial))

### Build a mask ##############################################################

# Thresholding
pvalues[pvalues < 5] = 0
plot_stat_map(nibabel.Nifti1Image(pvalues, fmri_img.get_affine()), mean_img,
              title='Thresholded p-values',
              cut_coords=(coronal, sagittal, axial))

# Binarization and intersection with VT mask
bin_pvalues = (pvalues != 0)
vt = nibabel.load(haxby_files.mask_vt[0]).get_data().astype(bool)
bin_pvalues_and_vt = np.logical_and(bin_pvalues, vt)
plot_roi(nibabel.Nifti1Image(bin_pvalues_and_vt.astype(np.int), 
                             fmri_img.get_affine()), 
         mean_img, title='Intersection with ventral temporal mask',
         cut_coords=(coronal, sagittal, axial))

# Dilation
from scipy import ndimage
dil_bin_pvalues_and_vt = ndimage.binary_dilation(bin_pvalues_and_vt)
plot_roi(nibabel.Nifti1Image(dil_bin_pvalues_and_vt.astype(np.int), 
                             fmri_img.get_affine()), 
         mean_img, title='Dilated mask', cut_coords=(coronal, sagittal, axial))

# Identification of connected components
labels, n_labels = ndimage.label(dil_bin_pvalues_and_vt)
plot_roi(nibabel.Nifti1Image(labels, fmri_img.get_affine()), 
         mean_img, title='Connected components',
         cut_coords=(coronal, sagittal, axial))
plt.show()

# Save the result
nibabel.save(nibabel.Nifti1Image(labels, fmri_img.get_affine()), 'mask.nii')
