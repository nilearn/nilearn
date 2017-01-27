"""
BIDS dataset first and second level analysis
============================================

Full step-by-step example of fitting a GLM to perform a first and second level
analysis in a BIDS dataset and visualizing the results. Details about the BIDS
standard can be consulted at https://openfmri.org/data-organization/

More specifically:

1. Download an fMRI BIDS dataset with two language conditions to contrast.
2. We extract automatically from the BIDS dataset first level model objects
3. We fit a second level model on the fitted first level models. Notice that
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

#############################################################################
# Fetch example BIDS dataset
# --------------------------
# We download a partial example BIDS dataset. It contains only the necessary
# information to run an statistical analysis using Nistats. The raw data
# subject folders only contain bold.json and events.tsv files, while the
# derivatives folder with preprocessed files contain preproc.nii and
# confounds.tsv files.
data_dir, _ = fetch_bids_langloc_dataset()

#############################################################################
# Obtain automatically FirstLevelModel objects and fit arguments
# --------------------------------------------------------------
# From the dataset directory we obtain automatically FirstLevelModel objects
# with their subject_id filled from the BIDS dataset. Moreover we obtain
# for each model a dictionary with run_imgs, events and confounder regressors
# since in this case a confounds.tsv file is available in the BIDS dataset.
# To get the first level models we only have to specify the dataset directory
# and the task_id as specified in the file names.
task_id = 'languagelocalizer'
models, fit_kwargs = first_level_models_from_bids(
    data_dir, task_id, preproc_space='MNI152nonlin2009aAsym',
    preproc_variant='smoothResamp')

#############################################################################
# Quick sanity check on fit arguments
# -----------------------------------
# Normally you might want to do some additional processing on event files
# or you might want to create or extend confound regressors. In this example
# dataset none of that is necessary, so we will just go through the
# experimental paradigm and check all is fine for the first subject.

############################################################################
# We just expect one run img per subject.
print([os.path.basename(run) for run in fit_kwargs[0]['run_imgs']])

##############################################################################
# The only confounds stored are regressors obtained from motion correction. As
# we can verify from the column headers of the confounds table corresponding
# to the only run_img present
print(fit_kwargs[0]['confounds'][0].columns)

############################################################################
# During this acquisition the subject read blocks of sentences and
# consonant strings. So these are our only two conditions in events.
# We verify there are n blocks for each condition.
print(fit_kwargs[0]['events'][0]['trial_type'].value_counts())

############################################################################
# First level model estimation
# ----------------------------
# Now we simply fit each first level model and plot for each subject the
# contrast that reveals the language network (language - string).
fig, axes = plt.subplots(nrows=2, ncols=5)
for midx, (model, model_kwargs) in enumerate(zip(models, fit_kwargs)):
    model.fit(**model_kwargs)
    zmap = model.compute_contrast('language-string')
    plotting.plot_glass_brain(zmap, colorbar=False, threshold=norm.isf(0.001),
                              title=('sub-' + model.subject_label),
                              axes=axes[midx / 5, midx % 5],
                              plot_abs=False, display_mode='x')
fig.suptitle('subjects z_map language netowrk (language - string)')
plt.show()

#########################################################################
# Second level model estimation
# -----------------------------
# We just have to provide the list of fitted FirstLevelModel objects
# to the SecondLevelModel object for estimation. We can do this since
# all subjects share the same design matrix.
first_level_conditions = [['language', 'language'], ['string', 'string']]
second_level_input = models
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(second_level_input,
                                            first_level_conditions)

#########################################################################
# Computing contrasts at the second level is similar to the first level
zmap = second_level_model.compute_contrast('language-string')

#########################################################################
# The group level contrast of the language network is mostly left
# lateralized as expected
plotting.plot_glass_brain(zmap, colorbar=False, threshold=norm.isf(0.001),
                          title='Group language network',
                          plot_abs=False, display_mode='x')
plotting.show()
