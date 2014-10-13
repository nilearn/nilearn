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
from nilearn.input_data import NiftiLabelsMasker


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
first_roi_data = (labels == 1).astype(np.int)
second_roi_data = (labels == 2).astype(np.int)
plot_roi(nibabel.Nifti1Image(first_roi_data, fmri_img.get_affine()),
         mean_img, title='Connected components: first ROI')
plot_roi(nibabel.Nifti1Image(second_roi_data, fmri_img.get_affine()),
         mean_img, title='Connected components: second ROI')
plot_roi(nibabel.Nifti1Image(first_roi_data, fmri_img.get_affine()),
         mean_img, title='Connected components: first ROI',
         output_file='snapshot_first_ROI.png')  # no plot, save .PNG to hard disk 
plot_roi(nibabel.Nifti1Image(second_roi_data, fmri_img.get_affine()),
         mean_img, title='Connected components: second ROI',
         output_file='snapshot_second_ROI.png')  # no plot, save .PNG to hard disk 

# use the new ROIs to extract data maps in first ROI (label=1)
plt.figure()
n_scans = 1000
first_maps = nibabel.Nifti1Image(fmri_data[..., :n_scans], fmri_img.get_affine())
masker = NiftiLabelsMasker(
            labels_img=nibabel.Nifti1Image(labels, fmri_img.get_affine()),
            resampling_target=None,
            standardize=False,
            detrend=False)
act_summaries = masker.fit_transform(first_maps)
bins = np.arange(1, n_scans + 1)
roi1_time_series = act_summaries[:, 0]
plt.plot(bins, roi1_time_series)
ymin, ymax = np.min(roi1_time_series), np.max(roi1_time_series)
plt.axis([1, n_scans, ymin, ymax])
plt.title('Data extracted from ROI 1')
plt.xlabel('first %i images in dataset' % n_scans)
plt.ylabel('average values')

plt.show()

# save the ROI 'atlas' to a single output nifti
nibabel.save(nibabel.Nifti1Image(labels, fmri_img.get_affine()),
    'mask_atlas.nii')
