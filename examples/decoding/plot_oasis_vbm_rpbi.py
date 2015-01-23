"""
Voxel-Based Morphometry on Oasis dataset
========================================

This example uses Voxel-Based Morphometry (VBM) to study the relationship
between aging and gray matter density.

The data come from the `OASIS <http://www.oasis-brains.org/>`_ project.
If you use it, you need to agree with the data usage agreement available
on the website.

It has been run through a standard VBM pipeline (using SPM8 and
NewSegment) to create VBM maps, which we study here.

We perform a standard analysis using mass-univariate GLM (here
permuted to have exact correction for multiple comparisons). We then
use Randomized Parcellation Based Inference [1] to attain more
sensitivity.

____

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
#          Gael Varoquaux, Apr 2014
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 50  # more subjects requires more memory and more time

### Load Oasis dataset ########################################################
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

### Preprocess data ###########################################################
nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=0,
    memory='nilearn_cache')  # cache options
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(gray_matter_map_filenames)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = gm_maps_masked.shape
print n_samples, "subjects, ", n_features, "features"

### Inference with massively univariate model #################################
from nilearn.mass_univariate import permuted_ols

print "Massively univariate model"
neg_log_pvals, all_scores, _ = permuted_ols(
    age, gm_maps_masked,  # + intercept as a covariate by default
    n_perm=1000,  # 1,000 for the sake of time. 10,000 is recommended
    two_sided_test=False,  # RPBI does not perform a two-sided test
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals))

### Show results
from nilearn.plotting import plot_stat_map
bg_filename = gray_matter_map_filenames[0]
z_slice = 0
from nilearn.image.resampling import coord_transform
affine = neg_log_pvals_unmasked.get_affine()
_, _, k_slice = coord_transform(0, 0, z_slice,
                                linalg.inv(affine))
k_slice = round(k_slice)
threshold = -np.log10(0.1)  # 10% corrected

fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

display = plot_stat_map(neg_log_pvals_unmasked, bg_img=bg_filename,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig)

neg_log_pvals_slice_data = neg_log_pvals_unmasked.get_data()[..., k_slice]
n_detections = (neg_log_pvals_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values'
         '\n(Non-parametric + max-type correction)'
         '\n%d detections') % n_detections
display.title(title, y=1.2)

### Randomized Parcellation Based Inference ###################################
from nilearn.mass_univariate import randomized_parcellation_based_inference

print "Randomized Parcellation Based Inference"
neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    age, gm_maps_masked,  # + intercept as a covariate by default
    np.asarray(nifti_masker.mask_img_.get_data()).astype(bool),
    n_parcellations=30,  # 30 for the sake of time, 100 is recommended
    n_parcels=1000,
    threshold='auto',
    n_perm=1000,  # 1,000 for the sake of time. 10,000 is recommended
    random_state=0, memory='nilearn_cache', n_jobs=1, verbose=True)
neg_log_pvals_rpbi_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_rpbi))

### Show results
fig = plt.figure(figsize=(5.5, 7.5), facecolor='k')

display = plot_stat_map(neg_log_pvals_rpbi_unmasked, bg_img=bg_filename,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig)

neg_log_pvals_rpbi_slice_data = \
    neg_log_pvals_rpbi_unmasked.get_data()[..., k_slice]
n_detections = (neg_log_pvals_rpbi_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values'
         '\n(RPBI)'
         '\n%d detections') % n_detections
display.title(title, y=1.2)

plt.show()
