"""Studying firts-level-model details in a trials-and-error fashion
================================================================

In this tutorial, we study the parametrization of the first-level
model used for fMRI data analysis and clarify their impact on the
results of the analysis.

We use an exploratory approach, in which we incrementally include some
new features in the analysis and look at the outcome, i.e. the
resulting brain maps.

Readers without prior experience in fMRI data analysis should first
run the plot_sing_subject_single_run tutorial to get a bit more
familiar with the base concepts, and only then run thi script.

To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

import numpy as np
import pandas as pd
from nilearn import plotting
from nistats.first_level_model import FirstLevelModel
from nistats import datasets

###############################################################################
# Retrieving the data
# -------------------
#
# We use a so-called localizer dataset, which consists in a 5-minutes
# acquisition of a fast event-related dataset.

subject_data = datasets.fetch_spm_multimodal_fmri()
tr = 2.
from nilearn.image import concat_imgs, mean_img, threshold_img, crop_img
fmri_img = concat_imgs(subject_data.func1, auto_resample=True)

#########################################################################
# Create mean image for display
mean_image = mean_img(fmri_img)
bg_image = crop_img(threshold_img(mean_image, 66))

#########################################################################
# Get the experimental paradigm
n_scans = fmri_img.shape[-1]
from scipy.io import loadmat
timing = loadmat(getattr(subject_data, "trials_ses1"),
                 squeeze_me=True, struct_as_record=False)
faces_onsets = timing['onsets'][0].ravel()
scrambled_onsets = timing['onsets'][1].ravel()
onsets = np.hstack((faces_onsets, scrambled_onsets))
onsets *= tr  # because onsets were reporting in 'scans' units
conditions = (['faces'] * len(faces_onsets) +
              ['scrambled'] * len(scrambled_onsets))
paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})

# Build design matrix
frame_times = np.arange(n_scans) * tr
from nistats.design_matrix import make_design_matrix
design_matrix = make_design_matrix(frame_times, paradigm)

#########################################################################
# We can specify some contrasts (To get corresponding maps)
# for the sake of script concision, it is advatageous to make it a function

def make_contrasts(design_matrix):
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])
    return{
        'faces-scrambled': contrasts['faces'] - contrasts['scrambled'],
        'scrambled-faces': -contrasts['faces'] + contrasts['scrambled'],
        'effects_of_interest': np.vstack((contrasts['faces'],
                                          contrasts['scrambled']))
    }

contrasts = make_contrasts(design_matrix)

#########################################################################
# Fit GLM
print('Fitting a GLM')
fmri_glm = FirstLevelModel(tr)
fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrix)

#########################################################################
# Compute contrast maps
#

from nilearn import plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 2.5))
for i, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    ax = plt.subplot(1, len(contrasts), i + 1)
    z_map = fmri_glm.compute_contrast(
        contrast_val, output_type='z_score')
    plotting.plot_stat_map(
        z_map, bg_img=bg_image, threshold=3.0, display_mode='z', vmax=7,
        black_bg=True, title=contrast_id, axes=ax, cut_coords=[0])

plotting.show()


#########################################################################
# let's explore now wome variants around this bbasic model
#
