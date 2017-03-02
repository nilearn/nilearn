"""
BIDS dataset first and second level analysis
============================================

Full step-by-step example of fitting a GLM to perform a first and second level
analysis in a BIDS dataset and visualizing the results. Details about the BIDS
standard can be consulted at http://bids.neuroimaging.io/

More specifically:

1. Download an fMRI BIDS dataset with two language conditions to contrast.
2. Extract automatically from the BIDS dataset first level model objects
3. Fit a second level model on the fitted first level models. Notice that
   in this case the preprocessed bold images were already normalized to the
   same MNI space.

Author : Martin Perez-Guevara: 2016
"""

import os
from nilearn import plotting
from scipy.stats import norm
import matplotlib.pyplot as plt

from nistats.datasets import fetch_bids_langloc_dataset
from nistats.first_level_model import first_level_models_from_bids
from nistats.second_level_model import SecondLevelModel

##############################################################################
# Fetch example BIDS dataset
# --------------------------
# We download a partial example BIDS dataset. It contains only the necessary
# information to run a statistical analysis using Nistats. The raw data
# subject folders only contain bold.json and events.tsv files, while the
# derivatives folder with preprocessed files contain preproc.nii and
# confounds.tsv files.
data_dir, _ = fetch_bids_langloc_dataset()

##############################################################################
# Obtain automatically FirstLevelModel objects and fit arguments
# --------------------------------------------------------------
# From the dataset directory we obtain automatically FirstLevelModel objects
# with their subject_id filled from the BIDS dataset. Moreover we obtain
# for each model a dictionary with run_imgs, events and confounder regressors
# since in this case a confounds.tsv file is available in the BIDS dataset.
# To get the first level models we only have to specify the dataset directory
# and the task_label as specified in the file names.
task_label = 'languagelocalizer'
space_label = 'MNI152nonlin2009aAsym'
models, models_run_imgs, models_events, models_confounds = \
    first_level_models_from_bids(
        data_dir, task_label, space_label,
        img_filters=[('variant', 'smoothResamp')])

#############################################################################
# Quick sanity check on fit arguments
# -----------------------------------
# Additional checks or information extraction from pre-processed data can
# be made here

############################################################################
# We just expect one run img per subject.
print([os.path.basename(run) for run in models_run_imgs[0]])

###############################################################################
# The only confounds stored are regressors obtained from motion correction. As
# we can verify from the column headers of the confounds table corresponding
# to the only run_img present
print(models_confounds[0][0].columns)

############################################################################
# During this acquisition the subject read blocks of sentences and
# consonant strings. So these are our only two conditions in events.
# We verify there are n blocks for each condition.
print(models_events[0][0]['trial_type'].value_counts())

############################################################################
# First level model estimation
# ----------------------------
# Now we simply fit each first level model and plot for each subject the
# contrast that reveals the language network (language - string).
fig, axes = plt.subplots(nrows=2, ncols=5)
model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
    model.fit(imgs, events, confounds)
    zmap = model.compute_contrast('language-string')
    plotting.plot_glass_brain(zmap, colorbar=False, threshold=norm.isf(0.001),
                              title=('sub-' + model.subject_label),
                              axes=axes[int(midx / 5), int(midx % 5)],
                              plot_abs=False, display_mode='x')
fig.suptitle('subjects z_map language network (unc p<0.001)')
plt.show()

#########################################################################
# Second level model estimation
# -----------------------------
# We just have to provide the list of fitted FirstLevelModel objects
# to the SecondLevelModel object for estimation. We can do this since
# all subjects share the same design matrix.
second_level_input = models
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(second_level_input)

#########################################################################
# Computing contrasts at the second level is as simple as at the first level
# Since we are not providing confounders we are performing an one-sample test
# at the second level with the images determined by the specified first level
# contrast.
zmap = second_level_model.compute_contrast(first_level_contrast='language-string')

#########################################################################
# The group level contrast of the language network is mostly left
# lateralized as expected
plotting.plot_glass_brain(zmap, colorbar=True, threshold=norm.isf(0.001),
                          title='Group language network (unc p<0.001)',
                          plot_abs=False, display_mode='x')
plotting.show()
