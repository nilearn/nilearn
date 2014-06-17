"""
Computing an ROI mask
=======================

Example showing how a T-test can be performed to compute an ROI
mask, and how simple operations can improve the quality of the mask
obtained.
"""
from nilearn.plotting.img_plotting import plot_epi, plot_stat_map, plot_roi

### Coordinates of the selected slice #########################################

coronal = 27
sagittal = 21
axial = 26

### Load the data #############################################################

# Fetch the data files from Internet
from nilearn import datasets
haxby_files = datasets.fetch_haxby(n_subjects=1)


# First load the fMRI data
import nibabel
fmri_img = nibabel.load(haxby_files.func[0])
fmri_data = fmri_img.get_data()
fmri_affine = fmri_img.get_affine()

# Second, load the labels
import numpy as np
labels = np.genfromtxt(haxby_files.session_target[0], skip_header=1,
                          usecols=[0], dtype=basestring)

### Visualization function ####################################################

import matplotlib.pyplot as plt

# One-liner to display brain slices
def plot_brain(brain, x, y, z, title, cmap='hot'):
    plt.figure(figsize=(7, 4))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    brain = brain.copy()
    plt.imshow(brain[x, :, :].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Coronal')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(brain[:, y, :].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Sagittal')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(brain[:, :, z].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Axial')
    plt.suptitle(title)
    plt.tight_layout()


### Find voxels of interest ###################################################

# Smooth the data
from nilearn import image
fmri_img = image.smooth_img(haxby_files.func[0], fwhm=6)
fmri_data = fmri_img.get_data()

# Plot the mean image
mean_img = image.mean_img(fmri_img)
plot_epi(mean_img, title='Smoothed mean EPI')

# Run a T-test for face and houses
from scipy import stats
_, pvalues = stats.ttest_ind(fmri_data[..., labels == 'face'],
                             fmri_data[..., labels == 'house'], axis=-1)

# Use a log scale for p-values
pvalues = - np.log10(pvalues)
pvalues[np.isnan(pvalues)] = 0.
pvalues[pvalues > 10] = 10
plot_stat_map(nibabel.Nifti1Image(pvalues, fmri_img.get_affine()), mean_img,
              title="p-values")

### Build a mask ##############################################################

# Thresholding
plot_stat_map(nibabel.Nifti1Image(pvalues, fmri_img.get_affine()), mean_img, 
              threshold = 5, title='Thresholded p-values')

# Binarization and intersection with VT mask
bin_pvalues = (pvalues != 0)
vt = nibabel.load(haxby_files.mask_vt[0]).get_data().astype(bool)
bin_pvalues_and_vt = np.logical_and(pvalues, vt)
plot_roi(nibabel.Nifti1Image(bin_pvalues_and_vt.astype(np.int), 
                             fmri_img.get_affine()), 
         mean_img, title='Intersection with ventral temporal mask')

# Dilation
from scipy import ndimage
dil_bin_pvalues_and_vt = ndimage.binary_dilation(bin_pvalues_and_vt)
plot_roi(nibabel.Nifti1Image(dil_bin_pvalues_and_vt.astype(np.int), 
                             fmri_img.get_affine()), 
         mean_img, title='Dilated mask')

# Identification of connected components
labels, n_labels = ndimage.label(bin_pvalues_and_vt)
plot_roi(nibabel.Nifti1Image(labels, fmri_img.get_affine()), 
         mean_img, title='Connected components')
plt.show()

# Save the result
nibabel.save(nibabel.Nifti1Image(labels, fmri_affine), 'mask.nii')
