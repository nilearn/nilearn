"""
Computing an ROI mask
=======================

Example showing how a T-test can be performed to compute an ROI
mask, and how simple operations can improve the quality of the mask
obtained.
"""

##############################################################################
# Coordinates of the slice we will be displaying

coronal = -24
sagittal = -33
axial = -17
cut_coords = (coronal, sagittal, axial)

##############################################################################
# Load the data

# Fetch the data files from Internet
from nilearn import datasets
from nilearn.image import new_img_like


haxby_dataset = datasets.fetch_haxby(n_subjects=1)

# print basic information on the dataset
print('First subject anatomical nifti image (3D) located is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is located at: %s' %
      haxby_dataset.func[0])

# Second, load the labels
import numpy as np

session_target = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
haxby_labels = session_target['labels']

import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker

##############################################################################
# Build a statistical test to find voxels of interest

# Smooth the data
from nilearn import image
fmri_filename = haxby_dataset.func[0]
fmri_img = image.smooth_img(fmri_filename, fwhm=6)

# Plot the mean image
from nilearn.plotting import plot_epi
mean_img = image.mean_img(fmri_img)
plot_epi(mean_img, title='Smoothed mean EPI', cut_coords=cut_coords)

##############################################################################
# Run a T-test for face and houses
from scipy import stats
fmri_data = fmri_img.get_data()
_, p_values = stats.ttest_ind(fmri_data[..., haxby_labels == b'face'],
                              fmri_data[..., haxby_labels == b'house'],
                              axis=-1)

# Use a log scale for p-values
log_p_values = -np.log10(p_values)
log_p_values[np.isnan(log_p_values)] = 0.
log_p_values[log_p_values > 10.] = 10.
from nilearn.plotting import plot_stat_map
plot_stat_map(new_img_like(fmri_img, log_p_values),
              mean_img, title="p-values", cut_coords=cut_coords)

##############################################################################
# Build a mask from this statistical map

# Thresholding
log_p_values[log_p_values < 5] = 0
plot_stat_map(new_img_like(fmri_img, log_p_values),
              mean_img, title='Thresholded p-values', annotate=False,
              colorbar=False, cut_coords=cut_coords)

##############################################################################
# Binarization and intersection with VT mask
# (intersection corresponds to an "AND conjunction")
bin_p_values = (log_p_values != 0)
mask_vt_filename = haxby_dataset.mask_vt[0]
import nibabel
vt = nibabel.load(mask_vt_filename).get_data().astype(bool)
bin_p_values_and_vt = np.logical_and(bin_p_values, vt)

from nilearn.plotting import plot_roi, show
plot_roi(new_img_like(fmri_img, bin_p_values_and_vt.astype(np.int)),
         mean_img, title='Intersection with ventral temporal mask',
         cut_coords=cut_coords)

##############################################################################
# Dilation
from scipy import ndimage
dil_bin_p_values_and_vt = ndimage.binary_dilation(bin_p_values_and_vt)
plot_roi(new_img_like(fmri_img, dil_bin_p_values_and_vt.astype(np.int)),
         mean_img, title='Dilated mask', cut_coords=cut_coords,
         annotate=False)


##############################################################################
# Identification of connected components
labels, n_labels = ndimage.label(dil_bin_p_values_and_vt)
first_roi_data = (labels == 1).astype(np.int)
second_roi_data = (labels == 2).astype(np.int)
plot_roi(new_img_like(fmri_img, first_roi_data),
         mean_img, title='Connected components: first ROI')

plot_roi(new_img_like(fmri_img, second_roi_data),
         mean_img, title='Connected components: second ROI')


##############################################################################
# Use the new ROIs to extract data maps in both ROIs
masker = NiftiLabelsMasker(
    labels_img=new_img_like(fmri_img, labels),
    resampling_target=None,
    standardize=False,
    detrend=False)
masker.fit()
condition_names = list(set(haxby_labels))
n_cond_img = fmri_data[..., haxby_labels == b'house'].shape[-1]
n_conds = len(condition_names)

X1, X2 = np.zeros((n_cond_img, n_conds)), np.zeros((n_cond_img, n_conds))
for i, cond in enumerate(condition_names):
    cond_maps = new_img_like(
        fmri_img, fmri_data[..., haxby_labels == cond][..., :n_cond_img])
    mask_data = masker.transform(cond_maps)
    X1[:, i], X2[:, i] = mask_data[:, 0], mask_data[:, 1]
condition_names[condition_names.index(b'scrambledpix')] = b'scrambled'


##############################################################################
# Plot the average in the different condition names
plt.figure(figsize=(15, 7))
for i in np.arange(2):
    plt.subplot(1, 2, i + 1)
    plt.boxplot(X1 if i == 0 else X2)
    plt.xticks(np.arange(len(condition_names)) + 1, condition_names,
               rotation=25)
    plt.title('Boxplots of data in ROI%i per condition' % (i + 1))

show()

# save the ROI 'atlas' to a single output Nifti
nibabel.save(new_img_like(fmri_img, labels),
             'mask_atlas.nii')
