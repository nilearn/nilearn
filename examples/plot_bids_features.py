"""
The power of BIDS standards with an openneuro dataset
=======================================================

Full step-by-step example of fitting a GLM to perform a first level analysis
in an openneuro BIDS dataset. We demonstrate how BIDS derivatives, BIDS models
and NIDM json export can be exploited to perform a simple analysis with minimal
code and end with automatic publication that facilitates sharing and repproduction.
Details about the BIDS standard can be consulted at http://bids.neuroimaging.io/
We also demonstrate how to download individual groups of files from the
Openneuro s3 bucket.

More specifically:

1. Download an fMRI BIDS dataset with derivatives from openneuro
2. Extract automatically from the BIDS dataset first level model objects
3. Demonstrate Quality assurance of Nistat estimation against available FSL estimation in the openneuro dataset
4. Display contrast plot and uncorrected first level statistics table report

Author : Martin Perez-Guevara: 2017
"""

import os
from nilearn import plotting
from nilearn.input_data import NiftiMasker
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib

# from nistats.datasets import fetch_openneuro_dataset
from nistats.first_level_model import first_level_models_from_bids
from nistats.reporting import (
    compare_niimgs, plot_contrast_matrix, get_clusters_table)

##############################################################################
# Fetch openneuro BIDS dataset
# -----------------------------
# We download one subject from an openneuro BIDS dataset.
# We consider the stopsignal task from the ds000030 V4 dataset.
# It contains the necessary information to run a statistical analysis using
# Nistats. Also statistical results from an FSL analysis for an example QA.

# exclusion_patterns = ['.*group.*', '.*phenotype.*', '.*mriqc.*',
#                       '.*parameter_plots.*', '.*physio_plots.*',
#                       '.*space-fsaverage.*', '.*space-T1w.*',
#                       '.*dwi.*', '.*beh.*', '.*task-bart.*',
#                       '.*task-rest.*', '.*task-scap.*', '.*task-task.*']
# data_dir, _ = fetch_openneuro_dataset(exclusion_filters=exclusion_patterns)
data_dir = '/home/mfpgt/nilearn_data/ds000030/ds000030_R1.0.4/uncompressed'

##############################################################################
# Obtain automatically FirstLevelModel objects and fit arguments
# ---------------------------------------------------------------
# From the dataset directory we obtain automatically FirstLevelModel objects
# with their subject_id filled from the BIDS dataset. Moreover we obtain
# for each model a dictionary with run_imgs, events and confounder regressors
# since in this case a confounds.tsv file is available in the BIDS dataset.
# To get the first level models we have to specify the dataset directory,
# the task_label and the space_label as specified in the file names.
# We also have to provide the folder with the desired derivatives, that in this
# were produced by the fmriprep BIDS app.
task_label = 'stopsignal'
space_label = 'MNI152NLin2009cAsym'
derivatives_folder = 'derivatives/fmriprep'
models, models_run_imgs, models_events, models_confounds = \
    first_level_models_from_bids(
        data_dir, task_label, space_label, smoothing_fwhm=5.0,
        derivatives_folder=derivatives_folder)

#############################################################################
# Take model and model arguments of the subject and process events
model, imgs, events, confounds = (
    models[0], models_run_imgs[0], models_events[0], models_confounds[0])

subject = 'sub-' + model.subject_label

fsl_design_matrix_path = os.path.join(
    data_dir, 'derivatives', 'task', subject, 'stopsignal.feat', 'design.mat')
design_matrix_file = open(fsl_design_matrix_path, 'r')
for line in design_matrix_file:
    if '/Matrix' in line:
        break
design_matrix = np.array([list(map(float, line.replace('\t\n', '').split('\t'))) for
                          line in design_matrix_file])
design_columns = ['cond_%02d' % i for i in range(design_matrix.shape[1])]
design_columns[0] = 'Go'
design_columns[4] = 'StopSuccess'
design_matrix = pd.DataFrame(design_matrix, columns=design_columns)

#############################################################################
# Construct StopSucess - Go contrast of the Stop Signal task
contrast = np.array([0] * design_matrix.shape[1])

############################################################################
# First level model estimation (one subject)
# -------------------------------------------
# We fit the first level model for one subject.
model.fit(imgs, design_matrices=[design_matrix])

#############################################################################
# Then we compute the StopSuccess - Go contrast
z_map = model.compute_contrast('StopSuccess - Go')

#############################################################################
# We show Nistats agreement with the FSL estimation available in openneuro
fsl_z_map = nib.load(
    os.path.join(data_dir, 'derivatives', 'task', subject, 'stopsignal.feat',
                 'stats', 'zstat12.nii.gz'))

plotting.plot_glass_brain(z_map, colorbar=True, threshold=norm.isf(0.001),
                          title='Nistats Z map of "StopSuccess - Go" (unc p<0.001)',
                          plot_abs=False, display_mode='ortho')
plotting.plot_glass_brain(fsl_z_map, colorbar=True, threshold=norm.isf(0.001),
                          title='FSL Z map of "StopSuccess - Go" (unc p<0.001)',
                          plot_abs=False, display_mode='ortho')
plt.show()

compare_niimgs([z_map], [fsl_z_map], model.masker_,
               ref_label='Nistats', src_label='FSL')
plt.show()

#############################################################################
# Simple statistical report of thresholded contrast
# -----------------------------------------------------
# We display the contrast plot and table with cluster information
plot_contrast_matrix('StopSuccess - Go', design_matrix)
plotting.plot_glass_brain(z_map, colorbar=True, threshold=norm.isf(0.001),
                          plot_abs=False, display_mode='z',
                          figure=plt.figure(figsize=(4, 4)))
plt.show()

print(get_clusters_table(z_map, norm.isf(0.001), 10).to_latex())
