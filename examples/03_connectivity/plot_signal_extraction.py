"""
Extracting signals from a brain parcellation
============================================

Here we show how to extract signals from a brain parcellation and compute
a correlation matrix.

We also show the importance of defining good confounds signals: the
first correlation matrix is computed after regressing out simple
confounds signals: movement regressors, white matter and CSF signals, ...
The second one demonstrated the functionality of the
:mod:`nilearn.load_confounds` module to select sensible confound variables
from fMRIprep outputs. The third one is without any confounds: all regions
are connected to each other.

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
# -------------------------------
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
# ---------------------------------------------------
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
# ----------------------------------------
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
# ----------------------------------------------------------------------
# The :mod:`nilearn.load_confounds` module can be used to create
# a :class:`nilearn.load_confounds.Confounds` class with flexible
# parameters. We create a :class:`nilearn.load_confounds.Confounds` class
# which specifies strategies and optional parameters. Then the
# :func:`nilearn.load_confounds.Confounds.load` method selects the relevant
# columns from the TSV file. Let's try a minimal strategy removing motion
# and global signal, and some of its variations:

from nilearn.load_confounds import Confounds
minimal = Confounds(strategy=["high_pass", "motion", "wm_csf", "global"],
                    motion="basic", wm_csf="basic", global_signal="basic")

confounds_minimal, sample_mask = minimal.load(fmri_filenames)

print("The shape of the confounds matrix is:", confounds_minimal.shape)
print(confounds_minimal.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_minimal,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Minimal', reorder=True)

# Now let's create a set with motion scrubbing
minimal_scrub = Confounds(strategy=["high_pass", "motion", "wm_csf", "scrub"],
                         motion="basic", wm_csf="basic", scrub="full")

confounds_minimal_scrub, sample_mask = minimal_scrub.load(fmri_filenames)

print("The shape of the confounds matrix is:", confounds_minimal_scrub.shape)
print(confounds_minimal_scrub.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_minimal_scrub,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Minimal and Scrubbing',
                     reorder=True)

# Now let's create a set of confounds without global signal
minimal_no_gsr = Confounds(strategy=["high_pass", "motion", "wm_csf"],
                           motion="basic", wm_csf="basic")

confounds_minimal_no_gsr, sample_mask = minimal_no_gsr.load(fmri_filenames)

print("The shape of the confounds matrix is:",
      confounds_minimal_no_gsr.shape)
print(confounds_minimal_no_gsr.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_minimal_no_gsr,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Minimal - no GSR',
                     reorder=True)

###############################################################################
# Extract signals and compute a connectivity matrix without confounds
# -------------------------------------------------------------------

time_series = masker.fit_transform(fmri_filenames)
# Note how we did not specify confounds above. This is bad!

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='No confounds', reorder=True)

plotting.show()
