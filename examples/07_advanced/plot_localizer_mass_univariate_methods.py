"""
Massively univariate analysis of a motor task from the Localizer dataset
========================================================================

This example shows the results obtained in a massively univariate
analysis performed at the inter-subject level with various methods.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.

1. A standard :term:`ANOVA` is performed. Data smoothed at 5
   :term:`voxels<voxel>` :term:`FWHM` are used.

2. A permuted Ordinary Least Squares algorithm is run at each :term:`voxel`.
   Data smoothed at 5 :term:`voxels<voxel>` :term:`FWHM` are used.


"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, May. 2014
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.image import get_data

##############################################################################
# Load Localizer contrast
n_samples = 94
localizer_dataset = datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'],
    n_subjects=n_samples, legacy_format=False
)

# print basic information on the dataset
print('First contrast nifti image (3D) is located at: %s' %
      localizer_dataset.cmaps[0])

tested_var = localizer_dataset.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
mask_quality_check = np.where(
    np.logical_not(np.isnan(tested_var))
)[0]
n_samples = mask_quality_check.size
contrast_map_filenames = [localizer_dataset.cmaps[i]
                          for i in mask_quality_check]
tested_var = tested_var[mask_quality_check].values.reshape((-1, 1))
print("Actual number of subjects after quality check: %d" % n_samples)


##############################################################################
# Mask data
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(contrast_map_filenames)


##############################################################################
# Anova (parametric F-scores)
from sklearn.feature_selection import f_regression
_, pvals_anova = f_regression(fmri_masked, tested_var, center=True)
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = - np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova)


##############################################################################
# Perform massively univariate analysis with permuted OLS
#
# This method will produce both voxel-level FWE-corrected -log10 p-values and
# :term:`TFCE`-based FWE-corrected -log10 p-values.
neg_log_pvals_permuted_ols, _, _, neg_log_pvals_tfce = permuted_ols(
    tested_var,
    fmri_masked,
    model_intercept=True,
    masker=nifti_masker,
    tfce=True,
    n_perm=1000,  # 1000 for the sake of time. Ideally, this should be 10000.
    verbose=1,  # display progress bar
    n_jobs=1,  # can be changed to use more CPUs
)
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_permuted_ols))
neg_log_pvals_tfce_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_tfce))


##############################################################################
# Visualization
from nilearn.plotting import plot_stat_map

# Various plotting parameters
z_slice = 12  # plotted slice
vmax = min(
    np.amax(neg_log_pvals_permuted_ols),
    np.amax(neg_log_pvals_anova),
    np.amax(neg_log_pvals_tfce),
)

threshold = - np.log10(0.1)  # 10% corrected

fig, axes = plt.subplots(figsize=(12, 3), facecolor='k', ncols=3)

# Plot Anova p-values
display = plot_stat_map(
    neg_log_pvals_anova_unmasked,
    threshold=threshold,
    display_mode='z',
    cut_coords=[z_slice],
    figure=fig,
    axes=axes[0],
    vmax=vmax,
    black_bg=True,
)

n_detections = (get_data(neg_log_pvals_anova_unmasked) > threshold).sum()
title = (
    'Negative $\\log_{10}$ p-values\n'
    '(Parametric +\n'
    'Bonferroni correction)\n'
    f'{n_detections} detections'
)

axes[0].set_title(title, color='white')

# Plot permuted OLS p-values
display = plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked,
    threshold=threshold,
    display_mode='z',
    cut_coords=[z_slice],
    figure=fig,
    axes=axes[1],
    vmax=vmax,
    black_bg=True,
)

n_detections = (
    get_data(neg_log_pvals_permuted_ols_unmasked) > threshold
).sum()
title = (
    'Negative $\\log_{10}$ p-values\n'
    '(Non-parametric +\n'
    'max-type correction)\n'
    f'{n_detections} detections'
)

axes[1].set_title(title, color='white')

# Plot permuted OLS TFCE-based p-values
display = plot_stat_map(
    neg_log_pvals_tfce_unmasked,
    threshold=threshold,
    display_mode='z',
    cut_coords=[z_slice],
    figure=fig,
    axes=axes[2],
    vmax=vmax,
    black_bg=True,
)

n_detections = (get_data(neg_log_pvals_tfce_unmasked) > threshold).sum()
title = (
    'Negative $\\log_{10}$ p-values\n'
    '(Non-parametric + \n'
    'TFCE max-type correction)\n'
    f'{n_detections} detections'
)

axes[2].set_title(title, color='white')

fig.show()
