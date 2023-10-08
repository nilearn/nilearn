"""
Extracting signals from a brain parcellation
============================================

Here we show how to extract signals from a brain parcellation and compute
a correlation matrix.

We also show the importance of defining good confounds signals: the
first correlation matrix is computed after regressing out simple
confounds signals: movement regressors, white matter and CSF signals, ...
The second one is without any confounds: all regions are connected to each
other. Finally we demonstrated the functionality of
:func:`nilearn.interfaces.fmriprep.load_confounds` to flexibly select confound
variables from :term:`fMRIPrep` outputs while following some implementation
guideline of :term:`fMRIPrep` confounds documentation
`<https://fmriprep.org/en/stable/outputs.html#confounds>`_.

One reference that discusses the importance of confounds is `Varoquaux and
Craddock, Learning and comparing functional connectomes across subjects,
NeuroImage 2013
<http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.

This is just a code example, see the :ref:`corresponding section in the
documentation <parcellation_time_series>` for more.

.. note::
    This example needs SciPy >= 1.0.0 for the reordering of the matrix.

.. include:: ../../../examples/masker_note.rst

"""


##############################################################################
# Retrieve the atlas and the data
# -------------------------------
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_filename = dataset.maps
labels = dataset.labels

print(f"Atlas ROIs are located in nifti image (4D) at: {atlas_filename}")

# One subject of brain development fMRI data
data = datasets.fetch_development_fmri(n_subjects=1, reduce_confounds=True)
fmri_filenames = data.func[0]
reduced_confounds = data.confounds[0]  # This is a preselected set of confounds

##############################################################################
# Extract signals on a :term:`parcellation` defined by labels
# -----------------------------------------------------------
# Using the NiftiLabelsMasker
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(
    labels_img=atlas_filename,
    standardize="zscore_sample",
    standardize_confounds="zscore_sample",
    memory="nilearn_cache",
    verbose=5,
)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(fmri_filenames, confounds=reduced_confounds)

##############################################################################
# Compute and display a correlation matrix
# ----------------------------------------
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(
    kind="correlation",
    standardize="zscore_sample",
)
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
plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Confounds",
    reorder=True,
)

##############################################################################
# Extract signals and compute a connectivity matrix without confounds removal
# ---------------------------------------------------------------------------
# After covering the basic of signal extraction and functional connectivity
# matrix presentation, let's look into the impact of confounds to :term:`fMRI`
# signal and functional connectivity. Firstly let's find out what a functional
# connectivity matrix looks like without confound removal.

time_series = masker.fit_transform(fmri_filenames)
# Note how we did not specify confounds above. This is bad!

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="No confounds",
    reorder=True,
)

##############################################################################
# Load confounds from file using a flexible strategy with fmriprep interface
# --------------------------------------------------------------------------
# The :func:`nilearn.interfaces.fmriprep.load_confounds` function provides
# flexible  parameters to retrieve the relevant columns from the TSV file
# generated by :term:`fMRIPrep`.
# :func:`nilearn.interfaces.fmriprep.load_confounds` ensures two things:
#
# 1. The correct regressors are selected with provided strategy, and
#
# 2. Volumes such as non-steady-state and/or high motion volumes are masked
#    out correctly.
#
# Let's try a simple strategy removing motion, white matter signal,
# cerebrospinal fluid signal with high-pass filtering.

from nilearn.interfaces.fmriprep import load_confounds

confounds_simple, sample_mask = load_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf"],
    motion="basic",
    wm_csf="basic",
)

print("The shape of the confounds matrix is:", confounds_simple.shape)
print(confounds_simple.columns)

time_series = masker.fit_transform(
    fmri_filenames, confounds=confounds_simple, sample_mask=sample_mask
)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Motion, WM, CSF",
    reorder=True,
)

##############################################################################
# Motion-based scrubbing
# ----------------------
# With a scrubbing-based strategy,
# :func:`~nilearn.interfaces.fmriprep.load_confounds` returns a `sample_mask`
# that removes the index of volumes exceeding the framewise displacement and
# standardised DVARS threshold, and all the continuous segment with less than
# five volumes. Before applying scrubbing, it's important to access the
# percentage of volumns scrubbed. Scrubbing is not a suitable strategy for
# datasets with too many high motion subjects.
# On top of the simple strategy above, let's add scrubbing to our
# strategy.

confounds_scrub, sample_mask = load_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf", "scrub"],
    motion="basic",
    wm_csf="basic",
    scrub=5,
    fd_threshold=0.2,
    std_dvars_threshold=3,
)

print(
    f"After scrubbing, {sample_mask.shape[0]} "
    f"out of {confounds_scrub.shape[0]} volumes remains"
)
print("The shape of the confounds matrix is:", confounds_simple.shape)
print(confounds_scrub.columns)

time_series = masker.fit_transform(
    fmri_filenames, confounds=confounds_scrub, sample_mask=sample_mask
)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Motion, WM, CSF, Scrubbing",
    reorder=True,
)

##############################################################################
# The impact of global signal removal
# -----------------------------------
# Global signal removes the grand mean from your signal. The benefit is that
# it can remove impacts of physiological artifacts with minimal impact on the
# degrees of freedom. The downside is that one cannot get insight into variance
# explained by certain sources of noise. Now let's add global signal to the
# simple strategy and see its impact.

confounds_minimal_no_gsr, sample_mask = load_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf", "global_signal"],
    motion="basic",
    wm_csf="basic",
    global_signal="basic",
)
print("The shape of the confounds matrix is:", confounds_minimal_no_gsr.shape)
print(confounds_minimal_no_gsr.columns)

time_series = masker.fit_transform(
    fmri_filenames, confounds=confounds_minimal_no_gsr, sample_mask=sample_mask
)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="Motion, WM, CSF, GSR",
    reorder=True,
)

##############################################################################
# Using predefined strategies
# ---------------------------
# Instead of customising the strategy through
# :func:`nilearn.interfaces.fmriprep.load_confounds`, one can use a predefined
# strategy with :func:`nilearn.interfaces.fmriprep.load_confounds_strategy`.
# Based on the confound variables generated through :term:`fMRIPrep`, and past
# benchmarks studies (:footcite:`Ciric2017`, :footcite:`Parker2018`): `simple`,
# `scrubbing`, `compcor`, `ica_aroma`.
# The following examples shows how to use the `simple` strategy and overwrite
# the motion default to basic.

from nilearn.interfaces.fmriprep import load_confounds_strategy

# use default parameters
confounds, sample_mask = load_confounds_strategy(
    fmri_filenames, denoise_strategy="simple", motion="basic"
)
time_series = masker.fit_transform(
    fmri_filenames, confounds=confounds, sample_mask=sample_mask
)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="simple",
    reorder=True,
)

# add optional parameter global signal
confounds, sample_mask = load_confounds_strategy(
    fmri_filenames,
    denoise_strategy="simple",
    motion="basic",
    global_signal="basic",
)
time_series = masker.fit_transform(
    fmri_filenames, confounds=confounds, sample_mask=sample_mask
)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(
    correlation_matrix,
    figure=(10, 8),
    labels=labels[1:],
    vmax=0.8,
    vmin=-0.8,
    title="simple with global signal",
    reorder=True,
)

plotting.show()

##############################################################################
# References
# ----------
#
# .. footbibliography::

# sphinx_gallery_dummy_images=2
