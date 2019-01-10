"""
Statistical testing of a second-level analysis
==============================================

Perform a one-sample t-test on a bunch of images (a.k.a. second-level analyis in fMRI) and threshold the resulting statistical map.

This example is based on the so-called localizer dataset.
It shows activation related to a mental computation task,
as opposed to narrative sentence reading/listening.

"""

#########################################################################
# Prepare some images for a simple t test
# ----------------------------------------
# This is a simple manually performed second level analysis
from nilearn import datasets
import numpy as np
n_samples = 20
localizer_dataset = datasets.fetch_localizer_calculation_task(
    n_subjects=n_samples)

#########################################################################
# Get the set of individual statstical maps (contrast estimates)
cmap_filenames = localizer_dataset.cmaps

#########################################################################
# Perform the second level analysis
# ----------------------------------
#
# First define a design matrix for the model. As the model is trivial (one-sample test), the design matrix is just one column with ones.
import pandas as pd
design_matrix = pd.DataFrame([1] * n_samples, columns=['intercept'])

#########################################################################
# Specify and estimate the model
from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel().fit(
    cmap_filenames, design_matrix=design_matrix)

#########################################################################
# Compute the only possible contrast: the one-sample test. Since there
# is only one possible contrast, we don't need to specify it in detail
z_map = second_level_model.compute_contrast(output_type='z_score')

##########################################################################
from nilearn.input_data import NiftiMasker
from scipy.stats import norm
masker = NiftiMasker(mask_strategy='background').fit(z_map)
stats = np.ravel(masker.transform(z_map))
n_voxels = np.size(stats)
pvals = 2 * norm.sf(np.abs(stats))
pvals_corr = np.minimum(1, pvals * n_voxels)
neg_log_pvals = - np.log10(pvals_corr)
neg_log_pvals_unmasked = masker.inverse_transform(neg_log_pvals)

#########################################################################
# Visualize the results
from nilearn import plotting
display = plotting.plot_stat_map(z_map, title='Raw z map')
plotting.plot_stat_map(
    neg_log_pvals_unmasked, cut_coords=display.cut_coords)
plotting.show()

###########################################################################
filenames = cmap_filenames
tested_var = np.asarray([1] * n_samples)

##############################################################################
# Mask data
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker().fit(z_map)
data_masked = nifti_masker.fit_transform(filenames)

##############################################################################
# Perform massively univariate analysis with permuted OLS
from nilearn.mass_univariate import permuted_ols
neg_log_pvals_permuted_ols, _, _ = permuted_ols(
    tested_var, data_masked,
    model_intercept=True,
    n_perm=1000,
    n_jobs=1)
neg_log_pvals_permuted_ols_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals_permuted_ols))

###########################################################################
#Let us plot the second level contrast at the computed thresholds
from nilearn import plotting
plotting.plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked,
    cut_coords=display.cut_coords)
plotting.show()
