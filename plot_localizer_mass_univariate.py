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
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Mar. 2014
import numpy as np
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

### Load Localizer calculation contrast #######################################
n_samples = 50
dataset_files = datasets.fetch_localizer_calculation_task(n_subjects=n_samples)
covariates = dataset_files.ext_vars['score_cal_complexe']
# Quality check / Remove subjects with bad tested variate
mask_quality_check = np.where(covariates != 'None')[0]
n_samples = mask_quality_check.size
gray_matter_maps = [dataset_files.cmaps[i] for i in mask_quality_check]
covariates = covariates[mask_quality_check].astype(float).reshape((-1, 1))
print("Actual number of subjects after quality check: %d" % n_samples)

### Mask data #################################################################
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(gray_matter_maps)

### Perform massively univariate analysis with permuted OLS ###################
tested_var = np.ones((n_samples, 1), dtype=float)  # intercept
neg_log_pvals, _, _ = permuted_ols(
    tested_var, fmri_masked, confounding_vars=covariates,
    model_intercept=False, n_perm=1000,
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals))

neg_log_pvals2, _, _ = permuted_ols(
    tested_var, fmri_masked, confounding_vars=None, model_intercept=False,
    n_perm=1000,
    n_jobs=-1)  # can be changed to use more CPUs
neg_log_pvals2_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals2))

### Visualization #############################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Here, we should use a structural image as a background, when available.

# Various plotting parameters
picked_slice = 30  # plotted slice
vmin = - np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals), np.amax(neg_log_pvals2))
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")

# Plot p-values of the model without covariates
ax = grid[0]
p_ma = np.ma.masked_less(neg_log_pvals2_unmasked.get_data(), vmin)
ax.imshow(np.rot90(nifti_masker.mask_img_.get_data()[..., picked_slice]),
          interpolation='nearest', cmap=plt.cm.gray)
ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title('Without covariates' + '\n' +
             r'Negative $\log_{10}$ p-values' + '\n(Non-parametric + ' +
             '\nmax-type correction)' +
             '\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

# Plot p-values of the model with covariates
ax = grid[1]
p_ma = np.ma.masked_less(neg_log_pvals_unmasked.get_data(), vmin)
ax.imshow(np.rot90(nifti_masker.mask_img_.get_data()[..., picked_slice]),
          interpolation='nearest', cmap=plt.cm.gray)
im = ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
               cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title('With covariates' + '\n' +
             r'Negative $\log_{10}$ p-values' + '\n(Non-parametric + '
             '\nmax-type correction)' +
             # divide number of detections by 9 to compensate for resampling
             '\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

grid[0].cax.colorbar(im)
plt.subplots_adjust(0.02, 0.03, .95, 0.83)
plt.show()
