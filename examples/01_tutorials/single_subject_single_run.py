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

The analyse described here is performed in the native space, directly on the
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
#           function. To input your own data, you will need to provide
#           a list of paths to your own files in the ``subject_data`` variable.

from nistats.datasets import fetch_spm_auditory
subject_data = fetch_spm_auditory()
print(subject_data.func)  # print the list of names of functional images 
 
###############################################################################
# We can display the first functional image and the subject's anatomy:
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
plot_img(subject_data.func[0])
plot_anat(subject_data.anat)

###############################################################################
# Next, we concatenate all the 3D EPI image into a single 4D image, 
# the we average them in order to create a background
# image that will be used to display the activations:

from nilearn.image import concat_imgs, mean_img
fmri_img = concat_imgs(subject_data.func)
mean_img = mean_img(fmri_img)

###############################################################################
# Specifying the experimental paradigm
# ------------------------------------
#
# We must provide now a description of the experiment, that is, define the
# timing of the auditory stimulation and rest periods. According to
# the documentation of the dataset, there were sixteen 42s-long blocks --- in
# which 6 scans were acquired --- alternating between rest and
# auditory stimulation, starting with rest.
#
# The following table provide all the relevant informations: 
#

"""
duration,  onset,  trial_type
    42  ,    0  ,  rest
    42  ,   42  ,  active
    42  ,   84  ,  rest
    42  ,  126  ,  active
    42  ,  168  ,  rest
    42  ,  210  ,  active
    42  ,  252  ,  rest
    42  ,  294  ,  active
    42  ,  336  ,  rest
    42  ,  378  ,  active
    42  ,  420  ,  rest
    42  ,  462  ,  active
    42  ,  504  ,  rest
    42  ,  546  ,  active
    42  ,  588  ,  rest
    42  ,  630  ,  active
"""

# We can read such a table from a spreadsheet file  created with OpenOffice Calcor Office Excel, and saved under the *comma separated values* format (``.csv``).  

import pandas as pd
events = pd.read_csv('auditory_block_paradigm.csv')
print(events)

# ## ###################################################################
# # Alternatively, we could have used standard python
# # functions to create a pandas.DataFrame object that specifies the
# # timings:


###############################################################################
# Performing the GLM analysis
# ---------------------------
#
# It is now time to create and estimate a ``FirstLevelModel`` object, which will# generate the *design matrix* using the  information provided by the ``events` object.

from nistats.first_level_model import FirstLevelModel

fmri_glm = FirstLevelModel(t_r=7,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           period_cut=160)
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
# TODO: simplify!!!



# contrasts::
# TODO explain why contrasts are specified in such a way. Contrasts as weighted sum... use an example ....
 conditions =  {
  'active':            array([ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
  'rest':              array([ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]),
}
###############################################################################
# We can then compare the two conditions 'active' and 'rest' by
# generating the relevant contrast:

active_minus_rest = conditions['active'] - conditions['rest']

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

