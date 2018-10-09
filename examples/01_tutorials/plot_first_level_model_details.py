"""Studying first-level-model details in a trials-and-error fashion
================================================================

In this tutorial, we study the parametrization of the first-level
model used for fMRI data analysis and clarify their impact on the
results of the analysis.

We use an exploratory approach, in which we incrementally include some
new features in the analysis and look at the outcome, i.e. the
resulting brain maps.

Readers without prior experience in fMRI data analysis should first
run the :ref:`plot_single_subject_single_run` tutorial to get a bit more
familiar with the base concepts, and only then run this tutorial example.

To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

###############################################################################
# Retrieving the data
# -------------------
#
# We use a so-called localizer dataset, which consists in a 5-minutes
# acquisition of a fast event-related dataset.
#
from nistats import datasets
data = datasets.fetch_localizer_first_level()
fmri_img = data.epi_img

###############################################################################
# Define the paradigm that will be used. Here, we just need to get the provided file.
#
# This task, described in Pinel et al., BMC neuroscience 2007 probes
# basic functions, such as button with the left or right hand, viewing
# horizontal and vertical checkerboards, reading and listening to
# short sentences, and mental computations (subractions).
#
#  Visual stimuli were displayed in four 250-ms epochs, separated by
#  100ms intervals (i.e., 1.3s in total). Auditory stimuli were drawn
#  from a recorded male voice (i.e., a total of 1.6s for motor
#  instructions, 1.2-1.7s for sentences, and 1.2-1.3s for
#  subtraction). The auditory or visual stimuli were shown to the
#  participants for passive viewing or button response in
#  event-related paradigms.  Post-scan questions verified that the
#  experimental tasks were understood and followed correctly.
# 
# This task comprises 10 conditions:
#
# * clicGaudio: Left-hand three-times button press, indicated by visual instruction
# * clicDaudio: Right-hand three-times button press, indicated by visual instruction
# * clicGvideo: Left-hand three-times button press, indicated by auditory instruction
# * clicDvideo:  Right-hand three-times button press, indicated by auditory instruction
# * damier_H: Visualization of flashing horizontal checkerboards
# * damier_V: Visualization of flashing vertical checkerboards
# * phraseaudio: Listen to narrative sentences
# * phrasevideo: Read narrative sentences
# * calculaudio: Mental subtraction, indicated by auditory instruction
# * calculvideo: Mental subtraction, indicated by visual instruction
#

t_r = 2.4
events_file = data['events']
import pandas as pd
events= pd.read_table(events_file)

###############################################################################
# Running a basic model
# ---------------------
#
# First specify a linear model.
# the fit() model creates the design matrix and the beta maps.
#
from nistats.first_level_model import FirstLevelModel
first_level_model = FirstLevelModel(t_r)
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]

#########################################################################
# Let us take a look at the design matrix: it has 10 main columns corresponding to 10 experimental conditions, followed by 3 columns describing low-frequency signals (drifts) and a constant regressor.
from nistats.reporting import plot_design_matrix
plot_design_matrix(design_matrix)
import matplotlib.pyplot as plt
plt.show()

#########################################################################
# Specification of the contrasts.
# 
# For this, let's create a function that, given the design matrix,
# generates the corresponding contrasts.  This will be useful to
# repeat contrast specification when we change the design matrix.
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

#########################################################################
# So let's look at these computed contrasts

contrasts = make_localizer_contrasts(design_matrix)
plt.figure(figsize=(5, 9))
from nistats.reporting import plot_contrast_matrix
for i, (key, values) in enumerate(contrasts.items()):
    ax = plt.subplot(5, 1, i + 1)
    plot_contrast_matrix(values, design_matrix=design_matrix, ax=ax)

plt.show()

#########################################################################
# Contrast estimation and plotting
#
# Since this script will be repeated several times, for the sake of readability,
# we encapsulate it in a function that we call when needed.
#
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
# Let's run the model and look at the outcome.

plot_contrast(first_level_model)
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
# Note that the design matrix has more columns to model drifts in the data.
#
# We notice however that this model performs rather poorly.
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

first_level_model = FirstLevelModel(t_r, drift_model='polynomial',
                                    drift_order=5)
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
# This is the filter used to convert the event sequence into a
# reference BOLD signal for the design matrix.
#
# The first thing that we can do is to change the default model (the
# so-called Glover hrf) for the so-called canonical model of SPM
# --which has slightly weaker undershoot component.
 
first_level_model = FirstLevelModel(t_r, hrf_model='spm')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# No strong --positive or negative-- effect.
#
# We could try to go one step further: using not only the so-called
# canonical hrf, but also its time derivative. Note that in that case,
# we still perform the contrasts and obtain statistical significance
# for the main effect ---not the time derivative. This means that the
# inclusion of time derivative in the design matrix has the sole
# effect of discounting timing misspecification from the error term,
# which vould decrease the estimated variance and enhance the
# statistical significance of the effect. Is it the case ?

first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# Not a huge effect, but rather positive overall. We could keep that one.
#
# Bzw, a benefit of this approach is that we can test which voxels are
# well explined by the derivative term, hinting at misfit regions, a
# possibly valuable information This is implemented by an F test on
# the time derivative regressors.

contrast_val = np.eye(design_matrix.shape[1])[1:2:21]
z_map = first_level_model.compute_contrast(
    contrast_val, output_type='z_score')
plotting.plot_stat_map(
    z_map, display_mode='z', threshold=3.0, title='effect of time derivatives')
plt.show()

#########################################################################
# We don't see too much here: the onset times and hrf delay we're
# using are probably fine.

#########################################################################
# We can also consider adding the so-called dispersion derivative to
# capture some mis-specification in the shape of the hrf.
#
# This is done by specifying `hrf_model='spm + derivative + dispersion'`
#
first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative + dispersion')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# Not a huge effect. For the sake of simplicity and readibility, we
# can drop that one.

#########################################################################
# The noise model ar(1) or ols ?
# ------------------------------
#
# So far,we have implicitly used a lag-1 autoregressive model ---aka
# ar(1)--- for the temporal structure of the noise. An alternative
# choice is to use an ordinaly least squares model (ols) that assumes
# no temporal structure (time-independent noise)

first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative', noise_model='ols')
first_level_model = first_level_model.fit(fmri_img, events=events)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
# While the difference is not obvious you should rather stick to the ar(1) model, which is arguably more accurate.

#########################################################################
# Removing confounds
# ------------------
#
# A problematic feature of fMRI is the presence of unconctrolled
# confounds in the data, sue to scanner instabilities (spikes) or
# physiological phenomena, such as motion, heart and
# respiration-related blood oxygenation flucturations.  Side
# measurements are sometimes acquired to charcterise these
# effects. Here we don't have access to those.  What we can do instead
# is to estimate confounding effects from the data themselves, using
# the compcorr approach, and take those into account in the model.
#
# For this we rely on the so-called :ref:`high_variance_confounds
# <https://nilearn.github.io/modules/generated/nilearn.image.high_variance_confounds.html>`
# routine of Nilearn.


from nilearn.image import high_variance_confounds
confounds = pd.DataFrame(high_variance_confounds(fmri_img, percentile=1))
first_level_model = FirstLevelModel(t_r, hrf_model='spm + derivative')
first_level_model = first_level_model.fit(fmri_img, events=events,
                                          confounds=confounds)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
#  Note the five additional columns in the design matrix
#
# The effect on activation maps is complex: auditory/visual effects
# are killed, probably because they were somewhat colinear to the
# confounds. On the other hand, some of the maps become cleaner (H-V,
# computation) after this addition.


#########################################################################
#  Smoothing
# ----------
#
# Smoothing is a regularization of the model. It has two benefits:
# decrease the noise level in images, and reduce the discrepancy
# between individuals. The drawback is that it biases the shape and
# position of activation.  We simply illustrate here the statistical
# gains.  We use a mild smoothing of 5mm full-width at half maximum
# (fwhm).

first_level_model = FirstLevelModel(
    t_r, hrf_model='spm + derivative', smoothing_fwhm=5).fit(
        fmri_img, events=events, confounds=confounds)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()

#########################################################################
#  The design is unchanged but the maps are smoother and more contrasted
# 

#########################################################################
#  Masking
# --------
# Masking consists in selecting the region of the image on which the model is run: it is useless to run it outside of the brain.
#
# The approach taken by FirstLeveModel is to estimate it from the fMRI
# data themselves when no mask is explicitly provided.  Since the data
# have been resampled into MNI space, we can use instead a mask of the
# grey matter in MNI space. The benefit is that it makes voxel-level
# comparisons easier across subjects and datasets, and removed
# non-grey matter regions, in which no BOLD signal is expected.  The
# downside is that the mask may not fit very well these particular
# data.

from nilearn.plotting import plot_roi 
from nilearn.datasets import fetch_icbm152_brain_gm_mask
icbm_mask = fetch_icbm152_brain_gm_mask()
data_mask = first_level_model.masker_.mask_img_
plt.figure(figsize=(16, 4))
ax = plt.subplot(121)
plot_roi(icbm_mask, title='ICBM mask', axes=ax)
ax = plt.subplot(122)
plot_roi(data_mask, title='Data-driven mask', axes=ax)
plt.show()

#########################################################################
#  Impact on the first-level model


first_level_model = FirstLevelModel(
    t_r, hrf_model='spm + derivative', smoothing_fwhm=5).fit(
        fmri_img, events=events, confounds=confounds)
design_matrix = first_level_model.design_matrices_[0]
plot_design_matrix(design_matrix)
plot_contrast(first_level_model)
plt.show()



#########################################################################
# Conclusion
# ----------
#
# Interestingly, the model used here seems quite resilient to
# manipulation of modeling parameters: this is reassuring. It shows
# that Nistats defaults ('cosine' drift, cutoff=128s, 'glover' hrf,
# ar(1) model) are actually reasonable.  Note that these conclusions
# are specific to this dataset and may vary with other ones.
