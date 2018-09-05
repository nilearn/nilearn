"""
Group analysis of a motor task from the Localizer dataset
=========================================================

This example shows the results obtained in a group analysis using a more
complex contrast than a one- or two-sample t test.
We use the [left button press (auditory cue)] task from the Localizer
dataset and seek association between the contrast values and a variate
that measures the speed of pseudo-word reading. No confounding variate
is included in the model.


"""
# Author: Virgile Fritsch, Bertrand Thirion, 2014 -- 2018
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting

##############################################################################
# Load Localizer contrast
n_samples = 94
localizer_dataset = datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'], n_subjects=n_samples)

# print basic information on the dataset
print('First contrast nifti image (3D) is located at: %s' %
      localizer_dataset.cmaps[0])

tested_var = localizer_dataset.ext_vars['pseudo']
# Quality check / Remove subjects with bad tested variate
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

from nistats.second_level_model import SecondLevelModel
model = SecondLevelModel(smoothing_fwhm=5.0)
model.fit(contrast_map_filenames, design_matrix=design_matrix)

##########################################################################
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = model.compute_contrast('fluency', output_type='z_score')

###########################################################################
# We threshold the second level contrast at uncorrected p < 0.001 and plot
from nistats.thresholding import map_threshold
_, threshold = map_threshold(z_map, threshold=.05, height_control='fdr')

plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True,
    title='Group-level association between motor activity \n'
    'and reading fluency (fdr<0.05')

plotting.show()
