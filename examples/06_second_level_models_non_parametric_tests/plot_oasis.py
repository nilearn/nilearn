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
# Computing the (corrected) p-values with parametric test
from nilearn.image import math_img
from nilearn.input_data import NiftiMasker
p_val = second_level_model.compute_contrast(second_level_contrast='age',
                                            output_type='p_value')
masker = NiftiMasker(mask_img=mask_img).fit(p_val)
n_voxel = np.size(masker.transform(p_val))
# Correcting the p-values for multiple testing and taking neg log
neg_log_pval = math_img("-np.log10(np.minimum(1,img*{}))".format(str(n_voxel)),
                        img=p_val)

###########################################################################
# Let us plot the second level contrast
from nilearn import plotting
cut_coords = [-4, 26]
display = plotting.plot_stat_map(
    neg_log_pval, colorbar=True, display_mode='z', cut_coords=cut_coords)
plotting.show()

##############################################################################
from nistats.second_level_model import non_parametric_inference
neg_log_pvals_permuted_ols_unmasked = \
    non_parametric_inference(gray_matter_map_filenames,
                             design_matrix=design_matrix,
                             second_level_contrast='age',
                             model_intercept=True, n_perm=1000,
                             two_sided_test=False, mask=mask_img,
                             smoothing_fwhm=2.0, n_jobs=1)

###########################################################################
# Let us plot the second level contrast
from nilearn import plotting
display = plotting.plot_stat_map(
    neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=5,
    display_mode='z', cut_coords=cut_coords)
plotting.show()

# The neg-log p-values obtained with non parametric testing are capped at 3
# since the number of permutations is 1e3.
# Otherwise it seems that the non parametric test produce more discoveries
# and is then more powerfull than the usual parametric procedure.
