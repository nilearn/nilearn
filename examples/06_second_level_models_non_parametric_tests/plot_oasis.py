"""Voxel-Based Morphometry on Oasis dataset
========================================

This example uses Voxel-Based Morphometry (VBM) to study the relationship
between aging, sex and gray matter density.

The data come from the `OASIS <http://www.oasis-brains.org/>`_ project.
If you use it, you need to agree with the data usage agreement available
on the website.

It has been run through a standard VBM pipeline (using SPM8 and
NewSegment) to create VBM maps, which we study here.

VBM analysis of aging
---------------------

We run a standard GLM analysis to study the association between age
and gray matter density from the VBM data. We use only 100 subjects
from the OASIS dataset to limit the memory usage.

Note that more power would be obtained from using a larger sample of subjects.
"""
# Authors: Bertrand Thirion, <bertrand.thirion@inria.fr>, July 2018
#          Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
#          Gael Varoquaux, Apr 2014


n_subjects = 100  # more subjects requires more memory

############################################################################
# Load Oasis dataset
# ------------------

from nilearn import datasets
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

###############################################################################
# Sex is encoded as 'M' or 'F'. make it a binary variable
sex = oasis_dataset.ext_vars['mf'] == b'F'

###############################################################################
# Print basic information on the dataset
print('First gray-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.gray_matter_maps[0])  # 3D data
print('First white-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.white_matter_maps[0])  # 3D data

###############################################################################
# Get a mask image: A mask of the  cortex of the ICBM template
gm_mask = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# Resample the images, since this mask has a different resolution
from nilearn.image import resample_to_img
mask_img = resample_to_img(
    gm_mask, gray_matter_map_filenames[0], interpolation='nearest')

#############################################################################
# Analyse data
# ------------
#
# First create an adequate design matrix with three columns: 'age',
# 'sex', 'intercept'.
import pandas as pd
import numpy as np
intercept = np.ones(n_subjects)
design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                             columns=['age', 'sex', 'intercept'])

#############################################################################
# Plot the design matrix
from nistats.reporting import plot_design_matrix
ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')

##########################################################################
# Specify and fit the second-level model when loading the data, we
# smooth a little bit to improve statistical behavior

from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask=mask_img)
second_level_model.fit(gray_matter_map_filenames,
                       design_matrix=design_matrix)

##########################################################################
# Estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast(second_level_contrast=[1, 0, 0],
                                            output_type='z_score')

##########################################################################
from nilearn.input_data import NiftiMasker
from scipy.stats import norm
masker = NiftiMasker(smoothing_fwhm=2.0, mask_img=mask_img).fit(z_map)
stats = np.ravel(masker.transform(z_map))
n_voxels = np.size(stats)
pvals = 2 * norm.sf(np.abs(stats))
pvals_corr = np.minimum(1, pvals * n_voxels)
neg_log_pvals = - np.log10(pvals_corr)
neg_log_pvals_unmasked = masker.inverse_transform(neg_log_pvals)

###########################################################################
# Then plot it
from nilearn import plotting
cut_coords = [-4, 26]
display = plotting.plot_stat_map(
    neg_log_pvals_unmasked, colorbar=True, display_mode='z',
    cut_coords=cut_coords)
plotting.show()

##############################################################################
filenames = gray_matter_map_filenames
tested_var = age

##############################################################################
# Mask data
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(smoothing_fwhm=2.0, mask_img=mask_img).fit(z_map)
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
#Let us plot the second level contrast
from nilearn import plotting
display = plotting.plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    display_mode='z', vmax=5,
    cut_coords=cut_coords)
plotting.show()
