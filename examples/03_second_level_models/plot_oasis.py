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
____

"""
# Authors: Bertrand Thirion, <bertrand.thirion@inria.fr>, July 2018
#          Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
#          Gael Varoquaux, Apr 2014
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 100  # more subjects requires more memory

############################################################################
# Load Oasis dataset
# -------------------
oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars['age'].astype(float)

# sex is encoded as 'M' or 'F'. make it a binary variable
sex = oasis_dataset.ext_vars['mf'] == b'F'

# print basic information on the dataset
print('First gray-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.gray_matter_maps[0])  # 3D data
print('First white-matter anatomy image (3D) is located at: %s' %
      oasis_dataset.white_matter_maps[0])  # 3D data

## get a mask image
gm_mask = datasets.fetch_icbm152_brain_gm_mask()

## Since this mask has a different resolution
from nilearn.image import resample_to_img
mask_img = resample_to_img(
    gm_mask, gray_matter_map_filenames[0], interpolation='nearest')

#############################################################################
# Analyse data
# ----------------

from nistats.second_level_model import SecondLevelModel
import pandas as pd

intercept = np.ones(n_subjects)
design_matrix = pd.DataFrame(np.vstack((age, sex, intercept)).T,
                             columns=['age', 'sex', 'intercept'])
# plot the design matrix
from nistats.reporting import plot_design_matrix
ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')
plt.tight_layout()

##########################################################################
# specify and fit the model

second_level_model = SecondLevelModel(smoothing_fwhm=2.0, mask=mask_img)
second_level_model.fit(gray_matter_map_filenames,
                       design_matrix=design_matrix)

##########################################################################
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast(second_level_contrast=[1, 0, 0],
                                            output_type='z_score')

###########################################################################
# We threshold the second level contrast at uncorrected p < 0.001 and plot
from nilearn import plotting
from nistats.thresholding import map_threshold
_, threshold = map_threshold(
    z_map, threshold=.05, height_control='fdr')

display = plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True, display_mode='z',
    cut_coords=[-4, 26],
    title='age effect on grey matter density (FDR < .05)')

###########################################################################
# Can also study the effect of sex

z_map = second_level_model.compute_contrast(second_level_contrast='sex',
                                            output_type='z_score')
_, threshold = map_threshold(
    z_map, threshold=.05, height_control='fdr')
plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True,
    title='sex effect on grey matter density (FDR < .05)')

plotting.show()
###########################################################################
# Note that there is no significant effect of sex on grey matter density.
