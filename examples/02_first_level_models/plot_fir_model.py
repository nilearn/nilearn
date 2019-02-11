""" Analysis of an fMRI dataset with a Finite Impule Response (FIR) model
=====================================================================

FIR models are used to estimate the hemodyamic response non-parametrically.
The example below shows that they're good to do statistical inference
even on fast event-related fMRI datasets.

Specifically, the so-called 'localizer' dataset proposed by Pinel is
downloaded, then analysed with a FIR model with 3 lags.
4 main contrasts are estimated.

"""

import pandas as pd
from nistats import datasets
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix, plot_contrast_matrix
import matplotlib.pyplot as plt

data = datasets.fetch_localizer_first_level()
fmri_img = data.epi_img
t_r = 2.4
events_file = data['events']
events = pd.read_table(events_file)

import numpy as np


def make_localizer_contrasts(design_matrix):
    """ returns a dictionary of four contrasts, given the design matrix"""

    # first generate canonical contrasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])

    # Add more complex contrasts
    contrasts['audio'] = (contrasts['clicDaudio']
                          + contrasts['clicGaudio']
                          + contrasts['calculaudio']
                          + contrasts['phraseaudio']
                          )
    contrasts['video'] = (contrasts['clicDvideo']
                          + contrasts['clicGvideo']
                          + contrasts['calculvideo']
                          + contrasts['phrasevideo']
                          )
    contrasts['computation'] = contrasts['calculaudio'] + contrasts['calculvideo']
    contrasts['sentences'] = contrasts['phraseaudio'] + contrasts['phrasevideo']

    # Short dictionary of more relevant contrasts
    contrasts = {
        'left-right': (contrasts['clicGaudio']
                       + contrasts['clicGvideo']
                       - contrasts['clicDaudio']
                       - contrasts['clicDvideo']
                       ),
        'H-V': contrasts['damier_H'] - contrasts['damier_V'],
        'audio-video': contrasts['audio'] - contrasts['video'],
        'computation-sentences': (contrasts['computation'] -
                                  contrasts['sentences']
                                  ),
    }
    return contrasts

from nilearn import plotting

def plot_contrast(first_level_model):
    """ Given a first model, specify, enstimate and plot the main contrasts"""
    design_matrix = first_level_model.design_matrices_[0]
    # Call the contrast specification within the function
    contrasts = make_localizer_contrasts(design_matrix)
    fig = plt.figure(figsize=(11, 3))
    # compute the per-contrast z-map
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        ax = plt.subplot(1, len(contrasts), 1 + index)
        z_map = first_level_model.compute_contrast(
            contrast_val, output_type='z_score')
        plotting.plot_stat_map(
            z_map, display_mode='z', threshold=3.0, title=contrast_id, axes=ax,
            cut_coords=1)



#########################################################################
# Next solution is to try fininte impulse reponse (FIR) models: we just say that the hrf is an arbitrary function that lags behind the stimulus onset.
# In the present case, given that the numbers of condition is high, we should use a simple FIR model.
# 
# Concretely, we set `hrf_model` to 'fir' and `fir_delays` to [3, 5, 7] (s)
"""
first_level_model = FirstLevelModel(t_r, hrf_model='spm')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix_ = first_level_model.design_matrices_[0]
"""

first_level_model = FirstLevelModel(t_r, hrf_model='fir', fir_delays=[1, 2, 3])
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)



#########################################################################
# We have to change the contrast specification. We characterize the BOLD reposne by the sum across the three time lags. It's a bit hairy, sorry, but this is the price to pay for flexibility... 

contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])
conditions = events.trial_type.unique()
for condition in conditions:
    contrasts[condition] = np.sum(
        [contrasts[name] for name in design_matrix.columns
         if name[:len(condition)] == condition], 0)

contrasts["audio"] = np.sum(
    [contrasts[name] for name in
     ["clicDaudio", "clicGaudio", "calculaudio", "phraseaudio"]], 0)
contrasts["video"] = np.sum(
    [contrasts[name] for name in
     ["clicDvideo", "clicGvideo", "calculvideo", "phrasevideo"]], 0)
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

contrasts = {
    "left-right": (contrasts["clicGaudio"] + contrasts["clicGvideo"]
                   - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
    "H-V": contrasts["damier_H"] - contrasts["damier_V"],
    "audio-video": contrasts["audio"] - contrasts["video"],
    "computation-sentences": (contrasts["computation"] -
                              contrasts["sentences"]),
    }

plot_contrast_matrix(contrasts['left-right'], design_matrix)

#########################################################################
# Take a breathe.
#
# We can now  proceed by estimating the contrasts and displaying them.


fig = plt.figure(figsize=(11, 3))
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    ax = plt.subplot(1, len(contrasts), 1 + index)
    z_map = first_level_model.compute_contrast(
        contrast_val, output_type='z_score')
    plotting.plot_stat_map(
        z_map, display_mode='z', threshold=3.0, title=contrast_id, axes=ax,
        cut_coords=1)

#########################################################################
# The result is not convincing to my eyes. Maybe we're asking a bit too much to a small dataset, with a relatively large number of experimental conditions!
#

plt.show()
