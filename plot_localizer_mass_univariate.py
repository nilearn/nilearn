"""
Massively univariate analysis of a computation task from the Localizer dataset
==============================================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to determine which voxels are specifically active when a healthy subject
performs a computation task as opposed to a sentence reading task.
The same analysis is performed with a confounding variate (a score that
measures the subject's performance on complex calculation).

The example shows the differences that exist between two similar models:
(i) a test of the intercept without confounding variate;
(ii) the same test with a confounding variate.
It shows the importance of carefully choosing the model variates as the
results largely depend on it.

We use permuted OLS to perform the analysis as it is the only tools that can
easily deal with covariates for now.


"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, May. 2014
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

### Load Localizer calculation contrast #######################################
n_samples = 94
dataset_files = datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'],
    n_subjects=n_samples)
tested_var = dataset_files.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
mask_quality_check = np.where(tested_var != 'None')[0]
n_samples = mask_quality_check.size
gray_matter_maps = [dataset_files.cmaps[i] for i in mask_quality_check]
tested_var = tested_var[mask_quality_check].astype(float).reshape((-1, 1))
#print("Actual number of subjects after quality check: %d" % n_samples)

### Mask data #################################################################
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(gray_matter_maps)

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
    n_perm=1000,  # Idealy, this should be 10,000
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_permuted_ols))

### Visualization #############################################################

# Here, we should use a structural image as a background, when available.

# Various plotting parameters
picked_slice = 21  # plotted slice
vmin = - np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals_permuted_ols),
           np.amax(neg_log_pvals_anova))

# Here, we should use a structural image as a background, when available.

# Plot Anova p-values
plt.figure(figsize=(5, 5))
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
plt.tight_layout()

# Plot permuted OLS p-values
plt.figure(figsize=(5, 5))
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
plt.tight_layout()

plt.show()
