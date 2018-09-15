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

from nistats import datasets

###############################################################################
# Retrieving the data
# -------------------
#
# We use a so-called localizer dataset, which consists in a 5-minutes
# acquisition of a fast event-related dataset.

data = datasets.fetch_localizer_first_level()
t_r = 2.4
paradigm_file = data.paradigm
events= pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None)
events.columns = ['session', 'trial_type', 'onset']
fmri_img = data.epi_img

###############################################################################
# Running a basic model
# ---------------------

from nistats.first_level_model import FirstLevelModel
first_level_model = FirstLevelModel(t_r)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]

from nistats.reporting import plot_design_matrix
plot_design_matrix(design_matrix)

#########################################################################
# Specify the contrasts.
# 
# For this, let's create a function that, given the deisgn matrix,
# generates the corresponding contrasts.
# This will be useful

def make_localizer_contrasts(design_matrix):
    """ returns a dictionary of four contasts, given the design matrix"""
    # first generate canonical contrasts 
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = dict([(column, contrast_matrix[i])
                      for i, column in enumerate(design_matrix.columns)])
    # Add more complex contrasts
    contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
                         contrasts["calculaudio"] + contrasts["phraseaudio"]
    contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
                         contrasts["calculvideo"] + contrasts["phrasevideo"]
    contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
    contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

    #########################################################################
    # Short list of more relevant contrasts
    contrasts = {
        "left-right": (contrasts["clicGaudio"] + contrasts["clicGvideo"]
                       - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
        "H-V": contrasts["damier_H"] - contrasts["damier_V"],
        "audio-video": contrasts["audio"] - contrasts["video"],
        "computation-sentences": (contrasts["computation"] -
                                  contrasts["sentences"]),
    }
    return contrasts

contrasts = make_localizer_contrasts(design_matrix)
# TODO: plot contrasts

#########################################################################
# contrast estimation and plotting
# Since this script will be repeated several times, for the sake of readbility,
# we encapsulate it in a function that we call when needed.
#

import matplotlib.pyplot as plt

def plot_contrast(first_level_model):
    """ Given a first model, specify, enstimate and plot the main contrasts"""
    design_matrix = first_level_model.design_matrices_[0]
    # Call the contrast specification within the function
    contrasts = make_localizer_contrasts(design_matrix)
    fig = plt.figure(figsize=(11, 3))
    for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
        ax = plt.subplot(1, len(contrasts), 1 + index)
        z_map = first_level_model.compute_contrast(
            contrast_val, output_type='z_score')
        plotting.plot_stat_map(
            z_map, display_mode='z', threshold=3.0, title=contrast_id, axes=ax,
            cut_coords=1)

plt.show()

#########################################################################
# Changing the drift model
# ------------------------
# 
# By default the drift model is a set of slow oscillating functions (Discrete Cosine transform), with a cutoff at frequency 1/128 hz.
# We can change this cut-off, e.g. to 1/64Hz.
# This is done by setting period_cut=64(s)

first_level_model = FirstLevelModel(t_r, period_cut=64)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)

#########################################################################
# Does the model perform worse or better ?

plot_contrast(first_level_model)
plt.show()

#########################################################################
# Note that the design matrix has more columns to model dirft terms
#
# Anyway, this model performs rather poorly
#
# Another solution is to remove these drift terms. Maybe they're simply useless.
# this is done by setting drift_model to None.

first_level_model = FirstLevelModel(t_r, drift_model=None)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# Is it better than the original ? No !
#
# Note that the design matrix has changed with no drift columns.
# the event columns, on the other hand, haven't changed.
# Another alternative to get a drift model is to specify a set of polynomials
# Let's take a basis of 5 polynomials

first_level_model = FirstLevelModel(t_r, drift_model='polynomial', drift_order=5)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# Is it good ? No better, no worse. Let's turn to another parameter.

#########################################################################
# Changing the hemodynamic response model
# ---------------------------------------
#
# This is the filter used to convert the event sequence into a reference BOLD signal for the design matrix.
#
# The first thing that we can do is to change the default model (the so-called Glover hrf) for the so-called canocial model of SPM --which has slightly weaker undershoot component.  
 
first_level_model = FirstLevelModel(t_r, hrf_model='spm')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# No strong --positive or negative-- effect.
#
# We could try to go one step further: using not only the so-called canocial hrf, but also its time derivative. Note that in that case, we still perform the contrasts  and obtain statistical significance for the main effect ---not the time derivative. This means that the inclusion of time derivative in the design matrix has the sole effect of discounting timing misspecification from the error term, which vould decrease the estimated variance and enhance the statistical significance of the effect. Is it the case ?

first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# Not a huge effect, but rather positive overall. We could keep that one
#
# Bzw, a benefit of this approach is that we can test which voxels are well explined by the derivative term, hinting at misfit regions, a possibly valuable information
# This is implemented by an F test on the time derivative regressors.

contrast_val = np.eye(design_matrix.shape[1])[1:2:21]
z_map = first_level_model.compute_contrast(
    contrast_val, output_type='z_score')
plotting.plot_stat_map(
    z_map, display_mode='z', threshold=3.0, title="effect of time derivatives")
plt.show()

#########################################################################
# We don't see too much here: the onset times and hrf delay we're using are probably fine.

#########################################################################
# But we could also add the so-called dispersion derivative, to dicount hrf shape differeneces. Let's give a try
# Once again, we're only going to test the main regressor, not the dispersion derivative. This new model only changes the variance term.

first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative + dispersion')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# There are some milde effects. Maybe not worth the complexity increase !
# 
# Next solution is to try fininte impulse reponse (FIR) models: we just say that the hrf is an arbitrary function that lags behind the stimulus onset.
# In the present case, given that the numbers of condition is high, we should use a simple FIR model.
# 
# Concretely, we set `hrf_model` to 'fir' and `fir_delays` to [3, 5, 7] (s)


first_level_model = FirstLevelModel(t_r, hrf_model='fir', fir_delays=[3, 5, 7])
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plt.show()

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


#########################################################################
# Conclusion
# ----------
#
# Interestingly, the model used here seems quite resilient to manipulation of modeling parameters: this is reassuing. It shows that Nistats defaults ('cosine' drift, cutoff=128s, 'glover' hrf) are actually reasonable.
