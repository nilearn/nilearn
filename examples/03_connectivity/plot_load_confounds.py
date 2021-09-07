"""
load_confounds
============================================

This notebook show how to extract signals from a brain parcellation and
compute a correlation matrix, using different denoising strategies using the
load_confounds package. This notebook is adapted from an `existing nilearn
tutorial
<https://nilearn.github.io/auto_examples/03_connectivity/plot_signal_extraction.html#sphx-glr-auto-examples-03-connectivity-plot-signal-extraction-py>`_.

Author: Michael W. Weiss and Hao-Ting Wang
"""

##############################################################################
# Retrieve the atlas and the data
# --------------------------------
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

print('Atlas ROIs are located in nifti image (4D) at: %s' %
      atlas_filename)  # 4D data

# One subject of brain development fmri data
data = datasets.fetch_development_fmri(n_subjects=1)
fmri_filenames = data.func[0]

##############################################################################
# Load confounds from file using a flexible strategy
# -----------------------------------------------------
# load_confounds can be used to create a Confounds class with flexible
# parameters. We create a Confounds class which specifies strategies and
# optional parameters. Then the load() method selects the relevant columns from
# the TSV file. Let's try a strategy similar to Params9, but without the global
# signal regression:

from nilearn.load_confounds import Confounds
conf = Confounds(strategy=["high_pass", "motion", "wm_csf"],
                 motion="basic", wm_csf="basic")

confounds_flexible, sample_mask = conf.load(fmri_filenames)

print("The shape of the confounds matrix is:", confounds_flexible.shape)
print(confounds_flexible.columns)
##############################################################################
# Extract signals on a parcellation defined by labels
# -----------------------------------------------------
# Using the NiftiLabelsMasker
from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_flexible,
                                   sample_mask=sample_mask)

##############################################################################
# Compute and display a correlation matrix
# -----------------------------------------
from nilearn.connectome import ConnectivityMeasure
correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Plot the correlation matrix
import numpy as np
from nilearn import plotting
# Make a large figure
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)
# The labels we have start with the background (0), hence we skip the
# first label
# matrices are ordered for block-like representation
plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='High pass, motion, WM, CSF',
                     reorder=True)

###############################################################################
# Same thing without confounds, to stress the importance of confounds
# --------------------------------------------------------------------

time_series = masker.fit_transform(fmri_filenames)
# Note how we did not specify confounds above. This is bad!

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='No confounds', reorder=True)

plotting.show()
