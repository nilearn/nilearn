"""
GLM fitting in second level fMRI
================================

Full step-by-step example of fitting a GLM to perform a second level analysis
in experimental data and visualizing the results.

More specifically:

1. A sequence of subject fMRI button press images is downloaded.
2. A two-sample t-test is applied to the brain maps
to see the effect of the contrast difference across subjects.
"""

import pandas as pd
from nilearn import plotting
from nilearn.datasets import fetch_localizer_contrasts

#########################################################################
# Fetch dataset
# --------------
# We download a list of left vs right button press contrasts from a
# localizer dataset.
n_subjects = 16
sample_vertical = fetch_localizer_contrasts(
    ["vertical checkerboard"], n_subjects, get_tmaps=True)
sample_horizontal = fetch_localizer_contrasts(
    ["horizontal checkerboard"], n_subjects, get_tmaps=True)

# What remains implicit here is that there is a one-to-one
# correspondence between the two sample: the first image of both
# samples comes from subject S1, the second from subject S2 etc.

############################################################################
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
second_level_input = sample_vertical['cmaps'] + sample_horizontal['cmaps']

# model the effect of conditions (sample 1 vs sample 2)
import numpy as np
condition_effect = np.hstack(([1] * n_subjects, [- 1] * n_subjects))

# model the subject effect: each subject is observed in sample 1 and sample 2
subject_effect = np.vstack((np.eye(n_subjects), np.eye(n_subjects)))
subjects = ['S%02d' % i for i in range(1, n_subjects + 1)]
design_matrix = pd.DataFrame(
    np.hstack((condition_effect[:, np.newaxis], subject_effect)),
    columns=['vertical vs horizontal'] + subjects)

# plot the design_matrix
from nistats.reporting import plot_design_matrix
plot_design_matrix(design_matrix)

# formally specify the analysis model and fit it
from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(
    second_level_input, design_matrix=design_matrix)

##########################################################################
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast('vertical vs horizontal',
                                            output_type='z_score')

###########################################################################
# We threshold the second level contrast and plot it
threshold = 3.1  # correponds to  p < .001, uncorrected
display = plotting.plot_glass_brain(
    z_map, threshold=threshold, colorbar=True, plot_abs=False,
    title='vertical vs horizontal checkerboard (unc p<0.001')

plotting.show()
