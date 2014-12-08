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
haxby_labels = np.genfromtxt(haxby_files.session_target[0], skip_header=1,
                          usecols=[0], dtype=basestring)

### Visualization function ####################################################

import matplotlib.pyplot as plt
from nilearn.plotting import plot_epi, plot_stat_map, plot_roi
from nilearn.input_data import NiftiLabelsMasker

plt.close('all')

### Find voxels of interest ###################################################

# Smooth the data
from nilearn import image
fmri_img = image.smooth_img(haxby_files.func[0], fwhm=6)

# Plot the mean image
fig_id = plt.subplot(2, 1, 1)
mean_img = image.mean_img(fmri_img)
plot_epi(mean_img, title='Smoothed mean EPI', cut_coords=(coronal, sagittal,
    axial), axes=fig_id)

# Run a T-test for face and houses
from scipy import stats
fmri_data = fmri_img.get_data()
_, p_values = stats.ttest_ind(fmri_data[..., haxby_labels == 'face'],
                              fmri_data[..., haxby_labels == 'house'],
                              axis=-1)

# Use a log scale for p-values
log_p_values = -np.log10(p_values)
log_p_values[np.isnan(log_p_values)] = 0.
log_p_values[log_p_values > 10.] = 10.
fig_id = plt.subplot(2, 1, 2)
plot_stat_map(nibabel.Nifti1Image(log_p_values, fmri_img.get_affine()), mean_img,
    title="p-values", cut_coords=(coronal, sagittal, axial), axes=fig_id)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)

### Build a mask ##############################################################
plt.figure()
fig_id = plt.subplot(3, 1, 1)
# Thresholding
log_p_values[log_p_values < 5] = 0
plot_stat_map(nibabel.Nifti1Image(log_p_values, fmri_img.get_affine()), mean_img,
              title='Thresholded p-values', annotate=False, colorbar=False,
              cut_coords=(coronal, sagittal, axial), axes=fig_id)

# Binarization and intersection with VT mask
# (intersection corresponds to an "AND conjunction")
bin_p_values = (log_p_values != 0)
vt = nibabel.load(haxby_files.mask_vt[0]).get_data().astype(bool)
bin_p_values_and_vt = np.logical_and(bin_p_values, vt)

fig_id = plt.subplot(3, 1, 2)
plot_roi(nibabel.Nifti1Image(bin_p_values_and_vt.astype(np.int), 
         fmri_img.get_affine()), 
         mean_img, title='Intersection with ventral temporal mask',
         cut_coords=(coronal, sagittal, axial), axes=fig_id)

# Dilation
fig_id = plt.subplot(3, 1, 3)
from scipy import ndimage
dil_bin_p_values_and_vt = ndimage.binary_dilation(bin_p_values_and_vt)
plot_roi(nibabel.Nifti1Image(dil_bin_p_values_and_vt.astype(np.int),
    fmri_img.get_affine()),
    mean_img, title='Dilated mask', cut_coords=(coronal, sagittal, axial),
    axes=fig_id, annotate=False)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)

# Identification of connected components
plt.figure()
labels, n_labels = ndimage.label(dil_bin_p_values_and_vt)
first_roi_data = (labels == 1).astype(np.int)
second_roi_data = (labels == 2).astype(np.int)
fig_id = plt.subplot(2, 1, 1)
plot_roi(nibabel.Nifti1Image(first_roi_data, fmri_img.get_affine()),
    mean_img, title='Connected components: first ROI', axes=fig_id)
fig_id = plt.subplot(2, 1, 2)
plot_roi(nibabel.Nifti1Image(second_roi_data, fmri_img.get_affine()),
    mean_img, title='Connected components: second ROI', axes=fig_id)
plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0)
plot_roi(nibabel.Nifti1Image(first_roi_data, fmri_img.get_affine()),
    mean_img, title='Connected components: first ROI_',
    output_file='snapshot_first_ROI.png')
plot_roi(nibabel.Nifti1Image(second_roi_data, fmri_img.get_affine()),
    mean_img, title='Connected components: second ROI',
    output_file='snapshot_second_ROI.png')

# use the new ROIs to extract data maps in both ROIs
masker = NiftiLabelsMasker(
            labels_img=nibabel.Nifti1Image(labels, fmri_img.get_affine()),
            resampling_target=None,
            standardize=False,
            detrend=False)
masker.fit()
condition_names = list(set(haxby_labels))
n_cond_img = fmri_data[...,haxby_labels == 'house'].shape[-1]
n_conds = len(condition_names)
X1, X2 = np.zeros((n_cond_img, n_conds)), np.zeros((n_cond_img, n_conds))
for i, cond in enumerate(condition_names):
    cond_maps = nibabel.Nifti1Image(
        fmri_data[..., haxby_labels == cond][..., :n_cond_img],
        fmri_img.get_affine())
    mask_data = masker.transform(cond_maps)
    X1[:, i], X2[:, i] = mask_data[:, 0], mask_data[:, 1]
condition_names[condition_names.index('scrambledpix')] = 'scrambled'

plt.figure(figsize=(15, 7))
for i in np.arange(2):
    plt.subplot(1, 2, i + 1)
    plt.boxplot(X1 if i == 0 else X2)
    plt.xticks(np.arange(len(condition_names)) + 1, condition_names,
        rotation=25)
    plt.title('Boxplots of data in ROI%i per condition' % (i + 1))

plt.show()

# save the ROI 'atlas' to a single output nifti
nibabel.save(nibabel.Nifti1Image(labels, fmri_img.get_affine()),
    'mask_atlas.nii')
