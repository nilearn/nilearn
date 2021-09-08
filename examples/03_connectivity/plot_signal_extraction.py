"""
Extracting signals from a brain parcellation
============================================

Here we show how to extract signals from a brain parcellation and compute
a correlation matrix.

We also show the importance of defining good confounds signals: the
first correlation matrix is computed after regressing out simple
confounds signals: movement regressors, white matter and CSF signals, ...
The second one demonstrated the functionality of `load_confounds` module to
select sensible confound variables from fMRIprep outputs.The third one is
without any confounds: all regions are connected to each other.


One reference that discusses the importance of confounds is `Varoquaux and
Craddock, Learning and comparing functional connectomes across subjects,
NeuroImage 2013
<http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.

This is just a code example, see the :ref:`corresponding section in the
documentation <parcellation_time_series>` for more.

.. note::
    This example needs SciPy >= 1.0.0 for the reordering of the matrix.
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
data = datasets.fetch_development_fmri(n_subjects=1, reduce_confounds=True)
fmri_filenames = data.func[0]
reduced_confounfs = data.confounds[0]  # This is a preselected set of confounds

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
time_series = masker.fit_transform(fmri_filenames, confounds=reduced_confounfs)

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
                     vmax=0.8, vmin=-0.8, reorder=True)

##############################################################################
# Load confounds from file using a flexible strategy with load_confounds
# -----------------------------------------------------
# load_confounds can be used to create a Confounds class with flexible
# parameters. We create a Confounds class which specifies strategies and
# optional parameters. Then the load() method selects the relevant columns from
# the TSV file. Let's try a strategy similar to Params9 from Ciric et al. 2017,
# and some of its variations:

from nilearn.load_confounds import Confounds
params9 = Confounds(strategy=["high_pass", "motion", "wm_csf", "global"],
                    motion="basic", wm_csf="basic", global_signal="basic")

confounds_params9, sample_mask = params9.load(fmri_filenames)

print("The shape of the confounds matrix is:", confounds_params9.shape)
print(confounds_params9.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_params9,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Params9', reorder=True)

# Now let's create a set with motion scrubbing
params9scrub = Confounds(strategy=["high_pass", "motion", "wm_csf", "scrub"],
                         motion="basic", wm_csf="basic", scrub="full")

confounds_params9scrub, sample_mask = params9scrub.load(fmri_filenames)

print("The shape of the confounds matrix is:", confounds_params9scrub.shape)
print(confounds_params9scrub.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_params9scrub,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Params9Scrub',
                     reorder=True)

# Now let's create a set of confounds without global signal
params9_no_gsr = Confounds(strategy=["high_pass", "motion", "wm_csf"],
                           motion="basic", wm_csf="basic")

confounds_params9_no_gsr, sample_mask = params9_no_gsr.load(fmri_filenames)

print("The shape of the confounds matrix is:",
      confounds_params9_no_gsr.shape)
print(confounds_params9_no_gsr.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_params9_no_gsr,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Params9 - no GSR',
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
