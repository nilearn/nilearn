"""
Massively univariate analysis of a computation task from the Localizer dataset
==============================================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to determine which voxels are specifically active when a healthy subject
performs a computation task as opposed to a sentence reading task.

Randomized Parcellation Based Inference [1] is also used so as to illustrate
that it conveys more sensitivity.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Mar. 2014
import numpy as np
from nilearn import datasets
from scipy import linalg
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate import randomized_parcellation_based_inference

### Load Localizer motor contrast #############################################
n_samples = 20
# localizer_dataset = datasets.fetch_localizer_calculation_task(
#     n_subjects=n_samples)
localizer_dataset = datasets.fetch_localizer_contrasts(
    ["calculation vs sentences"],
    n_subjects=n_samples)

### Mask data #################################################################
nifti_masker = NiftiMasker(
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(localizer_dataset.cmaps)

### Perform massively univariate analysis with permuted OLS ###################
tested_var = np.ones((n_samples, 1), dtype=float)  # intercept
neg_log_pvals, all_scores, h0 = permuted_ols(
    tested_var, fmri_masked, model_intercept=False,
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    two_sided_test=False,  # RPBI does not perform a two-sided test
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals))

### Randomized Parcellation Based Inference ###################################
neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    tested_var, fmri_masked,
    np.asarray(nifti_masker.mask_img_.get_data()).astype(bool),
    n_parcellations=30,  # 30 for the sake of time, 100 is recommended
    n_parcels=1000,
    threshold='auto',
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    random_state=0, memory='nilearn_cache', n_jobs=1, verbose=True)
neg_log_pvals_rpbi_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_rpbi)

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

# Here, we should use a structural image as a background, when available.

# Various plotting parameters
z_slice = 39  # plotted slice
from nilearn.image.resampling import coord_transform
affine = neg_log_pvals_unmasked.get_affine()
_, _, k_slice = coord_transform(0, 0, z_slice,
                                linalg.inv(affine))
k_slice = round(k_slice)

threshold = - np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals),
           np.amax(neg_log_pvals_rpbi))

# Plot permutation p-values map
fig = plt.figure(figsize=(5, 7), facecolor='k')

display = plot_stat_map(neg_log_pvals_unmasked,
                        threshold=threshold, cmap=plt.cm.autumn,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=vmax, black_bg=True)

neg_log_pvals_data = neg_log_pvals_unmasked.get_data()
neg_log_pvals_slice_data = \
    neg_log_pvals_data[..., k_slice]
n_detections = (neg_log_pvals_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values'
         '\n(Non-parametric + '
         '\nmax-type correction)'
         '\n%d detections') % n_detections

display.title(title, y=1.2)

# Plot RPBI p-values map
fig = plt.figure(figsize=(5, 7), facecolor='k')

display = plot_stat_map(neg_log_pvals_rpbi_unmasked,
                        threshold=threshold, cmap=plt.cm.autumn,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=vmax, black_bg=True)

neg_log_pvals_rpbi_data = \
    neg_log_pvals_rpbi_unmasked.get_data()
neg_log_pvals_rpbi_slice_data = \
    neg_log_pvals_rpbi_data[..., k_slice]
n_detections = (neg_log_pvals_rpbi_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values' + '\n(RPBI)'
         '\n%d detections') % n_detections

display.title(title, y=1.2)

plt.show()
