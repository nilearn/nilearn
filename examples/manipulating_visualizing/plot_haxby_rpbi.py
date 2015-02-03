"""
Massively univariate analysis of face vs house recognition (2)
==============================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to detemine whether or not it shows a different mean value under a
"face viewing" condition and a "house viewing" condition.

Randomized Parcellation Based Inference [1] is also used on the same data.
It yields a better recovery of the activations as the method is almost
equivalent to applying anisotropic smoothing to the data.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
from scipy import linalg
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate import randomized_parcellation_based_inference

### Load Haxby dataset ########################################################
haxby_dataset = datasets.fetch_haxby_simple()

### Mask data #################################################################
mask_filename = haxby_dataset.mask
nifti_masker = NiftiMasker(
    standardize=True,  # important for RPBI not to be biased by anatomy
    mask_img=mask_filename,
    memory='nilearn_cache', memory_level=1)  # cache options
func_filename = haxby_dataset.func
fmri_masked = nifti_masker.fit_transform(func_filename)

### Restrict to faces and houses ##############################################
conditions_encoded, sessions = np.loadtxt(
    haxby_dataset.session_target).astype("int").T
conditions = np.recfromtxt(haxby_dataset.conditions_target)['f0']
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
conditions_encoded = conditions_encoded[condition_mask]
fmri_masked = fmri_masked[condition_mask]

# We consider the mean image per session and per condition.
# Otherwise, the observations cannot be exchanged at random because
# a time dependence exists between observations within a same session.
n_sessions = np.unique(sessions).size
grouped_fmri_masked = np.empty((2 * n_sessions,  # two conditions per session
                                fmri_masked.shape[1]))
grouped_conditions_encoded = np.empty((2 * n_sessions, 1))

for s in range(n_sessions):
    session_mask = sessions[condition_mask] == s
    session_house_mask = np.logical_and(session_mask,
                                        conditions[condition_mask] == 'house')
    session_face_mask = np.logical_and(session_mask,
                                       conditions[condition_mask] == 'face')
    grouped_fmri_masked[2 * s] = fmri_masked[session_house_mask].mean(0)
    grouped_fmri_masked[2 * s + 1] = fmri_masked[session_face_mask].mean(0)
    grouped_conditions_encoded[2 * s] = conditions_encoded[
        session_house_mask][0]
    grouped_conditions_encoded[2 * s + 1] = conditions_encoded[
        session_face_mask][0]

### Perform massively univariate analysis with permuted OLS ###################
# We use a two-sided t-test to compute p-values, but we keep trace of the
# effect sign to add it back at the end and thus observe the signed effect
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    grouped_conditions_encoded, grouped_fmri_masked,
    # + intercept as a covariate by default
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    two_sided_test=False,  # RPBI does not perform a two-sided test
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals)

### Randomized Parcellation Based Inference ###################################
neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    grouped_conditions_encoded, grouped_fmri_masked,
    # + intercept as a covariate by default
    nifti_masker.mask_img_.get_data().astype(bool),
    n_parcellations=30,  # 30 for the sake of time, 100 is recommended
    n_parcels=1000,
    threshold='auto',
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    random_state=0, memory='nilearn_cache',
    n_jobs=1, verbose=True)
neg_log_pvals_rpbi_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_rpbi)

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

# Use the fMRI mean image as a surrogate of anatomical data
from nilearn import image
mean_fmri_img = image.mean_img(haxby_dataset.func)

# Various plotting parameters
z_slice = -17  # plotted slice
from nilearn.image.resampling import coord_transform
affine = neg_log_pvals_unmasked.get_affine()
_, _, k_slice = coord_transform(0, 0, z_slice,
                                linalg.inv(affine))
k_slice = round(k_slice)
threshold = -np.log10(0.1)  # 10% corrected
vmax = min(neg_log_pvals.max(), neg_log_pvals_rpbi.max())

# Plot permutation p-values map
fig = plt.figure(figsize=(4, 5.5), facecolor='k')

display = plot_stat_map(neg_log_pvals_unmasked, mean_fmri_img,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=vmax)

neg_log_pvals_data = neg_log_pvals_unmasked.get_data()
neg_log_pvals_slice_data = neg_log_pvals_data[..., k_slice, 0]
n_detections = (neg_log_pvals_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values'
         '\n(Non-parametric two-sided test'
         '\n+ max-type correction)'
         '\n%d detections') % n_detections

display.title(title, y=1.1)

# Plot RPBI p-values map
fig = plt.figure(figsize=(4, 5.5), facecolor='k')

display = plot_stat_map(neg_log_pvals_rpbi_unmasked, mean_fmri_img,
                        threshold=threshold, cmap=plt.cm.RdBu_r,
                        display_mode='z', cut_coords=[z_slice],
                        figure=fig, vmax=vmax)

neg_log_pvals_rpbi_data = neg_log_pvals_rpbi_unmasked.get_data()
neg_log_pvals_rpbi_slice_data = neg_log_pvals_rpbi_data[..., k_slice]
n_detections = (neg_log_pvals_rpbi_slice_data > threshold).sum()
title = ('Negative $\log_{10}$ p-values' + '\n(RPBI)'
         '\n%d detections') % n_detections

display.title(title, y=1.1)

plt.show()
