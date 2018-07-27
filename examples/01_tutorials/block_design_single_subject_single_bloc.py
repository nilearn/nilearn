"""
Analysis of a block design (stimulation vs rest), single session, single subject.
=================================================================================

In this tutorial, we compare the fMRI signal during periods of auditory
stimulation versus periods of rest, using a General Linear Model (GLM).

The dataset comes from an experiment conducted at the FIL by Geriant Rees
under the direction of Karl Friston. It is provided by FIL methods
group which develops the SPM software.

According to SPM documentation, 96 scans were acquired (RT=7s) in one session. THe paradigm consisted of alternating periods of stimulation and rest, lasting 42s each (that is, for 6 scans). The sesssion started with a rest block.
Auditory stimulation consisted of bi-syllabic words presented binaurally at a
rate of 60 per minute. The functional data starts at scan number 4, that is the 
image file ``fM00223_004``.

The whole brain BOLD/EPI images were acquired on a  2T Siemens
MAGNETOM Vision system. Each scan consisted of 64 contiguous
slices (64x64x64 3mm x 3mm x 3mm voxels). Acquisition of one scan took 6.05s, with the scan to scan repeat time (RT) set arbitrarily to 7s.

This analyse described here is performed in the native space, on the
original EPI scans without any spatial or temporal preprocessing.
(More sensitive results would likely be obtained on the corrected,
spatially normalized and smoothed images).


To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

import matplotlib.pyplot as plt

###############################################################################
# Retrieving the data
# -------------------
#
# .. note:: In this tutorial, we load the data using a data downloading
#           function.To input your own data, you will need to pass
#           a list of paths to your own files.

from nistats.datasets import fetch_spm_auditory
subject_data = fetch_spm_auditory()


###############################################################################
# We can list the filenames of the functional images
print(subject_data.func)

###############################################################################
# Display the first functional image:
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
plot_img(subject_data.func[0])

###############################################################################
# Display the subject's anatomical image:
plot_anat(subject_data.anat)


###############################################################################
# Next, we concatenate all the 3D EPI image into a single 4D image:

from nilearn.image import concat_imgs
fmri_img = concat_imgs(subject_data.func)

###############################################################################
# And we average all the EPI images in order to create a background
# image that will be used to display the activations:

from nilearn import image
mean_img = image.mean_img(fmri_img)

###############################################################################
# Specifying the experimental paradigm
# ------------------------------------
#
# We must provide a description of the experiment, that is, define the
# timing of the auditory stimulation and rest periods. According to
# the documentation of the dataset, there were 16 42s blocks --- in
# which 6 scans were acquired --- alternating between rest and
# auditory stimulation, starting with rest. We use standard python
# functions to create a pandas.DataFrame object that specifies the
# timings:

import numpy as np
tr = 7.
slice_time_ref = 0.
n_scans = 96
epoch_duration = 6 * tr  # duration in seconds
conditions = ['rest', 'active'] * 8
n_blocks = len(conditions)
duration = epoch_duration * np.ones(n_blocks)
onset = np.linspace(0, (n_blocks - 1) * epoch_duration, n_blocks)

import pandas as pd
events = pd.DataFrame(
    {'onset': onset, 'duration': duration, 'trial_type': conditions})

###############################################################################
# The ``events`` object contains the information for the design:
print(events)


###############################################################################
# Performing the GLM analysis
# ---------------------------
#
# We need to construct a *design matrix* using the timing information
# provided by the ``events`` object. The design matrix contains
# regressors of interest as well as regressors of non-interest
# modeling temporal drifts:

frame_times = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
period_cut = 4. * epoch_duration
hrf_model = 'glover + derivative'

###############################################################################
# It is now time to create a ``FirstLevelModel`` object
# and fit it to the 4D dataset (Fitting means that the coefficients of the
# model are estimated to best approximate data)

from nistats.first_level_model import FirstLevelModel

fmri_glm = FirstLevelModel(tr, slice_time_ref, noise_model='ar1',
                           standardize=False, hrf_model=hrf_model,
                           drift_model=drift_model, period_cut=period_cut)
fmri_glm = fmri_glm.fit(fmri_img, events)

###############################################################################
# One can inspect the design matrix (rows represent time, and
# columns contain the predictors):

from nistats.reporting import plot_design_matrix
design_matrix = fmri_glm.design_matrices_[0]
plot_design_matrix(design_matrix)
plt.show()

###############################################################################
# The first column contains the expected reponse profile of regions which are
# sensitive to the auditory stimulation.


plt.plot(design_matrix['active'])
plt.xlabel('scan')
plt.title('Expected Auditory Response')
plt.show()


###############################################################################
# Detecting voxels with significant effects
# -----------------------------------------
#
# To access the estimated coefficients (Betas of the GLM model), we
# created constrasts with a single '1' in each of the columns:

contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])

"""
contrasts::

  {
  'active':            array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'active_derivative': array([ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'constant':          array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.]),
  'drift_1':           array([ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'drift_2':           array([ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'drift_3':           array([ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.]),
  'drift_4':           array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.]),
  'drift_5':           array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.]),
  'drift_6':           array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]),
  'drift_7':           array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.]),
  'rest':              array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'rest_derivative':   array([ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])}
"""

###############################################################################
# We can then compare the two conditions 'active' and 'rest' by
# generating the relevant contrast:

active_minus_rest = contrasts['active'] - contrasts['rest']

eff_map = fmri_glm.compute_contrast(active_minus_rest,
                                    output_type='effect_size')

z_map = fmri_glm.compute_contrast(active_minus_rest,
                                  output_type='z_score')

###############################################################################
# Plot thresholded z scores map

plot_stat_map(z_map, bg_img=mean_img, threshold=3.0,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (Z>3)')
plt.show()

###############################################################################
# We can use ``nibabel.save`` to save the effect and zscore maps to the disk

import os
outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

import nibabel
from os.path import join
nibabel.save(z_map, join('results', 'active_vs_rest_z_map.nii'))
nibabel.save(eff_map, join('results', 'active_vs_rest_eff_map.nii'))

###############################################################################
#  Extract the signal from a voxel
#  -------------------------------
#
# We search for the voxel with the larger z-score and plot the signal
# (warning: this is "double dipping")


# Find the coordinates of the peak

from nibabel.affines import apply_affine
values = z_map.get_data()
coord_peaks = np.dstack(np.unravel_index(np.argsort(-values.ravel()),
                                         values.shape))[0, 0, :]
coord_mm = apply_affine(z_map.affine, coord_peaks)

###############################################################################
# We create a masker for the voxel (allowing us to detrend the signal)
# and extract the time course

from nilearn.input_data import NiftiSpheresMasker
mask = NiftiSpheresMasker([coord_mm], radius=3,
                          detrend=True, standardize=True,
                          high_pass=None, low_pass=None, t_r=7.)
sig = mask.fit_transform(fmri_img)

##########################################################
# Let's plot the signal and the theoretical response

plt.plot(frame_times, sig, label='voxel %d %d %d' % tuple(coord_mm))
plt.plot(design_matrix['active'], color='red', label='model')
plt.xlabel('scan')
plt.legend()
plt.show()
