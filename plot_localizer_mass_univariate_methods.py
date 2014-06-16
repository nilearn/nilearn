"""
Massively univariate analysis of a motor task from the Localizer dataset
========================================================================

This example shows the results obtained in a massively univariate
analysis performed at the inter-subject level with various methods.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.

1. A standard Anova is performed. Data smoothed at 5 voxels FWHM are used.

2. A permuted Ordinary Least Squares algorithm is run at each voxel. Data
   smoothed at 5 voxels FWHM are used.


"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, May. 2014
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

### Load Localizer contrast ###################################################
n_samples = 94
dataset_files = datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'], n_subjects=n_samples)
tested_var = dataset_files.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
mask_quality_check = np.where(tested_var != 'None')[0]
n_samples = mask_quality_check.size
contrast_maps = [dataset_files.cmaps[i] for i in mask_quality_check]
tested_var = tested_var[mask_quality_check].astype(float).reshape((-1, 1))
print("Actual number of subjects after quality check: %d" % n_samples)

### Mask data #################################################################
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(contrast_maps)

### Anova (parametric F-scores) ###############################################
from nilearn._utils.fixes import f_regression
_, pvals_anova = f_regression(fmri_masked, tested_var, center=True)
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = - np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova)

### Perform massively univariate analysis with permuted OLS ###################
neg_log_pvals_permuted_ols, _, _ = permuted_ols(
    tested_var, fmri_masked,
    model_intercept=True,
    n_perm=5000,  # 5,000 for the sake of time. Idealy, this should be 10,000
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_permuted_ols))

### Visualization #############################################################

# Various plotting parameters
picked_slice = 21  # plotted slice
vmin = - np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals_permuted_ols),
           np.amax(neg_log_pvals_anova))

# Here, we should use a structural image as a background, when available.

# Plot Anova p-values
plt.figure(figsize=(5.5, 5.5))
masked_pvals = np.ma.masked_less(neg_log_pvals_anova_unmasked.get_data(), vmin)
plt.imshow(np.rot90(nifti_masker.mask_img_.get_data()[:, :, picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray)
im = plt.imshow(np.rot90(masked_pvals[:, :, picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title(r'Negative $\log_{10}$ p-values'
          + '\n(Parametric + Bonferroni correction)'
          + '\n%d detections' % (~masked_pvals.mask[..., picked_slice]).sum())

# Plot permuted OLS p-values
plt.figure(figsize=(5.5, 5.5))
masked_pvals = np.ma.masked_less(
    neg_log_pvals_permuted_ols_unmasked.get_data(), vmin)
plt.imshow(np.rot90(nifti_masker.mask_img_.get_data()[:, :, picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray)
im = plt.imshow(np.rot90(masked_pvals[:, :, picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title(r'Negative $\log_{10}$ p-values'
          + '\n(Non-parametric + max-type correction)'
          + '\n%d detections' % (~masked_pvals.mask[..., picked_slice]).sum())

plt.show()
