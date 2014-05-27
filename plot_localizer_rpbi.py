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
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import (permuted_ols,
                                     randomized_parcellation_based_inference)

### Load Localizer motor contrast #############################################
n_samples = 20
dataset_files = datasets.fetch_localizer_calculation_task(n_subjects=n_samples)

### Mask data #################################################################
nifti_masker = NiftiMasker(
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(dataset_files.cmaps)

### Perform massively univariate analysis with permuted OLS ###################
tested_var = np.ones((n_samples, 1), dtype=float)  # intercept
neg_log_pvals, all_scores, h0 = permuted_ols(
    tested_var, fmri_masked, model_intercept=False,
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    two_sided_test=False,  # RPBI does not perform a two-sided test
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals))

### RPBI ######################################################################
neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    tested_var, fmri_masked,
    np.asarray(nifti_masker.mask_img_.get_data()).astype(bool),
    n_parcellations=30,  # 30 for the sake of time, 100 is recommended
    n_parcels=1000,
    threshold='auto',
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    random_state=0, memory='nilearn_cache', n_jobs=1, verbose=True)
neg_log_pvals_rpbi_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_rpbi))

### Visualization #############################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Here, we should use a structural image as a background, when available.

# Various plotting parameters
picked_slice = 30  # plotted slice
vmin = - np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals), np.amax(neg_log_pvals_rpbi))
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")

# Plot permutation p-values map
ax = grid[0]
masked_pvals = np.ma.masked_less(neg_log_pvals_unmasked.get_data(), vmin)
ax.imshow(np.rot90(nifti_masker.mask_img_.get_data()[..., picked_slice]),
          interpolation='nearest', cmap=plt.cm.gray)
im = ax.imshow(np.rot90(masked_pvals[..., picked_slice]),
               interpolation='nearest',
               cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values' + '\n(Non-parametric + '
             '\nmax-type correction)' +
             '\n%d detections' % (~masked_pvals.mask[..., picked_slice]).sum())
ax.axis('off')

# Plot RPBI p-values map
ax = grid[1]
masked_pvals = np.ma.masked_less(neg_log_pvals_rpbi_unmasked.get_data(), vmin)
ax.imshow(np.rot90(nifti_masker.mask_img_.get_data()[..., picked_slice]),
          interpolation='nearest', cmap=plt.cm.gray)
ax.imshow(np.rot90(masked_pvals[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values' + '\n(RPBI)'
             + '\n\n%d detections'
             % (~masked_pvals.mask[..., picked_slice]).sum())
ax.axis('off')

grid[0].cax.colorbar(im)
plt.subplots_adjust(0.02, 0.03, .95, 0.83)
plt.show()
