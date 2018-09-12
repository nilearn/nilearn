"""
Analysis of a single session, single subject fMRI dataset
=========================================================

In this tutorial, we compare the fMRI signal during periods of auditory
stimulation versus periods of rest, using a General Linear Model (GLM).

The dataset comes from an experiment conducted at the FIL by Geriant Rees
under the direction of Karl Friston. It is provided by FIL methods
group which develops the SPM software.

According to SPM documentation, 96 scans were acquired (repetition time TR=7s) in one session. The paradigm consisted of alternating periods of stimulation and rest, lasting 42s each (that is, for 6 scans). The sesssion started with a rest block.
Auditory stimulation consisted of bi-syllabic words presented binaurally at a
rate of 60 per minute. The functional data starts at scan number 4, that is the 
image file ``fM00223_004``.

The whole brain BOLD/EPI images were acquired on a  2T Siemens
MAGNETOM Vision system. Each scan consisted of 64 contiguous
slices (64x64x64 3mm x 3mm x 3mm voxels). Acquisition of one scan took 6.05s, with the scan to scan repeat time (TR) set arbitrarily to 7s.

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
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
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
# We must now provide a description of the experiment, that is, define the
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

###############################################################################
# We can read such a table from a spreadsheet file  created with OpenOffice Calcor Office Excel, and saved under the *comma separated values* format (``.csv``). 
import pandas as pd
events = pd.read_csv('auditory_block_paradigm.csv')
print(events)

###############################################################################
# Performing the GLM analysis
# ---------------------------
#
# It is now time to create and estimate a ``FirstLevelModel`` object, that will generate the *design matrix* using the  information provided by the ``events` object.

from nistats.first_level_model import FirstLevelModel

###############################################################################
# Parameters of the first-level model
#
# * t_r=7(s) is the time of repetition of acquisitions
# * noise_model='ar1' specifies the noise covariance model: a lag-1 dependence
# * standardize=False means that we do not want to rescale the time series to mean 0, variance 1
# * hrf_model='spm' means that we rely on the SPM "canonical hrf" model (without time or dispersion derivatives)
# * drift_model='cosine' means that we model the signal drifts as slow oscillating time functions
# * period_cut=160(s) defines the cutoff frequency (its inverse actually).

fmri_glm = FirstLevelModel(t_r=7,
                           noise_model='ar1',
                           standardize=False,
                           hrf_model='spm',
                           drift_model='cosine',
                           period_cut=160)

###############################################################################
# Now that we have specified the mdoel, we can run it on the fMRI image
fmri_glm = fmri_glm.fit(fmri_img, events)

###############################################################################
# One can inspect the design matrix (rows represent time, and
# columns contain the predictors):

from nistats.reporting import plot_design_matrix
design_matrix = fmri_glm.design_matrices_[0]

###############################################################################
# We have taken the first design matrix, because the model is meant
# for multiple runs

import matplotlib.pyplot as plt
plot_design_matrix(design_matrix)
plt.show()

###############################################################################
# The first column contains the expected reponse profile of regions which are
# sensitive to the auditory stimulation.
# Let's plot this first column

plt.plot(design_matrix['active'])
plt.xlabel('scan')
plt.title('Expected Auditory Response')
plt.show()

###############################################################################
# Detecting voxels with significant effects
# -----------------------------------------
#
# To access the estimated coefficients (Betas of the GLM model), we
# created constrast with a single '1' in each of the columns: The role of the contrast is to select some columns of the model --and potentially weight them-- to study the associated statistics. So in a nutshell, a contrast is a weigted  combination of the estimated effects.
# Here we can define canonical contrasts that just consider the two condition in isolation ---let's call them "conditions"--- then a contrast that makes the difference between these conditions.

from numpy import array
conditions = {
    'active': array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
    'rest':   array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
}

###############################################################################
# We can then compare the two conditions 'active' and 'rest' by
# defining the corresponding contrast:

active_minus_rest = conditions['active'] - conditions['rest']

###############################################################################
# below, we compute the estimated effect. It is in BOLD signal unit, but has no statistical guarantees, because it does not take into account the associated variance.

eff_map = fmri_glm.compute_contrast(active_minus_rest,
                                    output_type='effect_size')

###############################################################################
# In order to get statistical significance, we form a t-statistic, and directly convert is into z-scale. The z-scale means that the values are scaled to match a standard Gaussian distribution (mean=0, variance=1), across voxels, if there are now effects in reality.

z_map = fmri_glm.compute_contrast(active_minus_rest,
                                  output_type='z_score')

###############################################################################
# Plot thresholded z scores map
# we display it on top of the average functional image of the series (could be the anatomical image of the subject).
# we use arbitrarily a threshold of 3.0 in z-scale. We'll see later how to use corrected thresholds.
# we show to display 3 axial views: display_mode='z', cut_coords=3

plot_stat_map(z_map, bg_img=mean_img, threshold=3.0,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (Z>3)')
plt.show()

###############################################################################
# Statistical signifiance testing
# One should worry about the statistical validity of the procedure: here we used an arbitrary threshold of 3.0 but the threshold should provide some guarantees on the risk of false detections (aka type-1 errors in statistics). One first suggestion is to control the false positive rate (fpr) at a certain level, e.g. 0.001: 

from nistats.thresholding import map_threshold
_, threshold = map_threshold(z_map, threshold=.001, height_control='fpr')
print('Uncorrected p<0.001 threshold: %.3f' % threshold)
plot_stat_map(z_map, bg_img=mean_img, threshold=threshold,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (p<0.001)')
plt.show()

###############################################################################
# The problem is that with this you expect a fraction of 0.001 * n_voxels to show up while they're not active. A more conservative solution is to control the family wise errro rate, i.e. the probability of making ony one false detection, say at 5%. For that we use the so-called Bonferroni correction

_, threshold = map_threshold(z_map, threshold=.05, height_control='bonferroni')
print('Bonferroni-corrected, p<0.05 threshold: %.3f' % threshold)
plot_stat_map(z_map, bg_img=mean_img, threshold=threshold,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (p<0.05, corrected)')
plt.show()

###############################################################################
# This is quite conservative indeed ! 
# A popular alternative is to control the false discovery rate, i.e. the expected proportion of false discoveries among detections. This is called the false disovery rate

_, threshold = map_threshold(z_map, threshold=.05, height_control='fdr')
print('False Discovery rate = 0.05 threshold: %.3f' % threshold)
plot_stat_map(z_map, bg_img=mean_img, threshold=threshold,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (fdr=0.05)')
plt.show()

###############################################################################
# Finally people like to discard isolated voxels (aka "small clusters") from these images. It is possible to generate a thresholded map with small clusters removed by providing a cluster_threshold argument. here clusters smaller than 10 voxels will be discarded.

clean_map, threshold = map_threshold(
    z_map, threshold=.05, height_control='fdr', cluster_threshold=10)
plot_stat_map(clean_map, bg_img=mean_img, threshold=threshold,
              display_mode='z', cut_coords=3, black_bg=True,
              title='Active minus Rest (fdr=0.05), clusters > 10 voxels')
plt.show()



###############################################################################
# We can save the effect and zscore maps to the disk
# first create a directory where you want to write the images

import os
outdir = 'results'
if not os.path.exists(outdir):
    os.mkdir(outdir)

###############################################################################
# Then save the images in this directory
    
from os.path import join
z_map.to_filename(join(outdir, 'active_vs_rest_z_map.nii.gz'))
eff_map.to_filename(join(outdir, 'active_vs_rest_eff_map.nii.gz'))

###############################################################################
# Report the found positions in a table

from nistats.reporting import get_clusters_table
table = get_clusters_table(z_map, stat_threshold=threshold,
                           cluster_threshold=20)
print(table)

###############################################################################
# the table can be saved for future use

table.to_csv(join(outdir, 'table.csv'))
