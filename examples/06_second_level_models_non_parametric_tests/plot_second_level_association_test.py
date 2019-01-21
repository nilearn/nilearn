"""
Example of generic design in second-level models
================================================

This example shows the results obtained in a group analysis using a more
complex contrast than a one- or two-sample t test.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.


"""
# Author: Virgile Fritsch, Bertrand Thirion, 2014 -- 2018


##############################################################################
# Load Localizer contrast
from nilearn import datasets
n_samples = 94
localizer_dataset = datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'], n_subjects=n_samples)

##############################################################################
# print basic information on the dataset
print('First contrast nifti image (3D) is located at: %s' %
      localizer_dataset.cmaps[0])

##############################################################################
# Load the behavioral variable
tested_var = localizer_dataset.ext_vars['pseudo']
print(tested_var)

##############################################################################
# Quality check / Remove subjects with bad tested variate
import numpy as np
mask_quality_check = np.where(tested_var != b'None')[0]
n_samples = mask_quality_check.size
contrast_map_filenames = [localizer_dataset.cmaps[i]
                          for i in mask_quality_check]
tested_var = tested_var[mask_quality_check].astype(float).reshape((-1, 1))
print("Actual number of subjects after quality check: %d" % n_samples)

############################################################################
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
import pandas as pd
design_matrix = pd.DataFrame(
    np.hstack((tested_var, np.ones_like(tested_var))),
    columns=['fluency', 'intercept'])

###########################################################################
# Fit of the second-level model
from nistats.second_level_model import SecondLevelModel
model = SecondLevelModel(smoothing_fwhm=5.0)
model.fit(contrast_map_filenames, design_matrix=design_matrix)

##########################################################################
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = model.compute_contrast('fluency', output_type='z_score')

##########################################################################
from nilearn.input_data import NiftiMasker
from scipy.stats import norm
masker = NiftiMasker(smoothing_fwhm=5.0, mask_strategy='background').fit(z_map)
stats = np.ravel(masker.transform(z_map))
n_voxels = np.size(stats)
pvals = 2 * norm.sf(np.abs(stats))
pvals_corr = np.minimum(1, pvals * n_voxels)
neg_log_pvals = - np.log10(pvals_corr)
neg_log_pvals_unmasked = masker.inverse_transform(neg_log_pvals)

###########################################################################
#Let us plot the second level contrast at the computed thresholds
from nilearn import plotting
cut_coords = [38, -17, -3]
plotting.plot_stat_map(
    neg_log_pvals_unmasked,  # threshold=threshold,
    colorbar=True, cut_coords=cut_coords)
plotting.show()

##############################################################################
from nistats.second_level_model import non_parametric_inference
neg_log_pvals_permuted_ols_unmasked = \
    non_parametric_inference(contrast_map_filenames,
                             design_matrix=design_matrix,
                             contrast='fluency',
                             model_intercept=True, n_perm=1000,
                             two_sided_test=True, mask=None,
                             smoothing_fwhm=5.0, n_jobs=1)

###########################################################################
#Let us plot the second level contrast
from nilearn import plotting
threshold_pval = 0
plotting.plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked,
    colorbar=True, cut_coords=cut_coords)
plotting.show()
