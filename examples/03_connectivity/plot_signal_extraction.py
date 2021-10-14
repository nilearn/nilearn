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
:func:`nilearn.input_data.fmriprep_confounds` to flexibly select confound
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
reduced_confounds = data.confounds[0]  # This is a preselected set of confounds

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
time_series = masker.fit_transform(fmri_filenames, confounds=reduced_confounds)

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
                     vmax=0.8, vmin=-0.8, title="Preset",
                     reorder=True)

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

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='No confounds', reorder=True)

##############################################################################
# Load confounds from file using a flexible strategy with fmriprep_confounds
# --------------------------------------------------------------------------
# The :func:`nilearn.input_data.fmriprep_confounds` function provides flexible
# parameters to retrieve the relevant columns from the TSV file generated by
# :term:`fMRIPrep`. :func:`nilearn.input_data.fmriprep_confounds` ensures
# two things:
# 1. The correct regressors are selected with provided strategy, and
# 2. Volumes such as non-steady-state volumes are masked out correctly.
# Let's try a simple strategy removing motion, white matter signal,
# cerebrospinal fluid signal with high-pass filtering.

from nilearn.input_data import fmriprep_confounds
confounds_simple, sample_mask = fmriprep_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf"],
    motion="basic", wm_csf="basic")

print("The shape of the confounds matrix is:", confounds_simple.shape)
print(confounds_simple.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_simple,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8, title='Motion, WM, CSF',
                     reorder=True)

##############################################################################
# Motion-based scrubbing
# ----------------------
# With a scrubbing-based strategy, `fmriprep_confounds` returns a `sample_mask`
# that removes the index of volumes exceeding the framewise displacement and
# standardised DVARS threshold, and all the continuous segment with less than
# five volumes. Before applying scrubbing, it's important to access the
# percentage of volumns scrubbed. Scrubbing is not a suitable strategy for
# datasets with too many high motion subjects.
# On top of the simple strategy above, let's add scrubbing to our
# strategy.

confounds_scrub, sample_mask = fmriprep_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf", "scrub"],
    motion="basic", wm_csf="basic",
    scrub=5, fd_thresh=0.2, std_dvars_thresh=3)

print("After scrubbing, {} out of {} volumes remains".format(
    sample_mask.shape[0], confounds_scrub.shape[0]))
print("The shape of the confounds matrix is:", confounds_simple.shape)
print(confounds_scrub.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_scrub,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8,
                     title='Motion, WM, CSF, Scrubbing',
                     reorder=True)

##############################################################################
# The impact of global signal removal
# -----------------------------------
# Global signal removes the grand mean from your signal. The benefit is that
# it can remove impacts of physiological artifacts with minimal impact on the
# degrees of freedom. The downside is that one cannot get insight into variance
# explained by certain sources of noise. Now let's add global signal to the
# simple strategy and see its impact.

confounds_minimal_no_gsr, sample_mask = fmriprep_confounds(
    fmri_filenames,
    strategy=["high_pass", "motion", "wm_csf", "global"],
    motion="basic", wm_csf="basic", global_signal="basic")
print("The shape of the confounds matrix is:",
      confounds_minimal_no_gsr.shape)
print(confounds_minimal_no_gsr.columns)

time_series = masker.fit_transform(fmri_filenames,
                                   confounds=confounds_minimal_no_gsr,
                                   sample_mask=sample_mask)

correlation_matrix = correlation_measure.fit_transform([time_series])[0]

np.fill_diagonal(correlation_matrix, 0)

plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=labels[1:],
                     vmax=0.8, vmin=-0.8,
                     title='Motion, WM, CSF, GSR',
                     reorder=True)

plotting.show()

##############################################################################
# Using predefined strategies
# ---------------------------
# Instead of customising the strategy through
# :func:`nilearn.input_data.fmriprep_confounds`, one can use a predefined
# strategy with :func:`nilearn.input_data.fmriprep_confounds_strategy`. Based
# on the confound variables generated through :term:`fMRIPrep`, and past
# benchmarks studies (:footcite:`Ciric2017`, :footcite:`Parker2018`), we
# provide four preset strategies: `simple`, `scrubbing`, `compcor`, and
# `ica_aroma`.
#   - `simple`: high pass filtering, basic motion, basic white matter, basic
#     csf, and optionally global signal. This approach is commonly
#     used in resting state functional connectivity, described in
#     :footcite:`Fox2005`. With the global signal option, this approach
#     can remove confounds without compromising the temporal degrees of freedom.
#   - `scrubbing` high pass filtering, fully expanded motion, white matter, and
#     csf parameters, scrubbing high motion volumes
#     (framewise displacement > 0.2 mm, standardized DVARS threshold > 3),
#     and optionally global signal. This approach can reliably remove the
#     impact of high motion volumes in functional connectome, however, it
#     might not be suitable with subjects with high motion (more than 50%
#     timeseries flagged as high motion). One should adjust the threshold
#     based on the characteristics of the dataset, or remove high motion subjects from
#     the dataset.
#   - `compcor` high pass filtering, fully expanded motion parameters, and
#     anatomical compcor components with combined white matter and csf mask
#     that fits 50% of the variance. Compcor can suffer from loss of
#     temporal degrees of freedom when using explained variance as the noise
#     component estimation. CompCor has the advantage of removing
#     physiological noise without requiring external monitoring of
#     physiological fluctuations (:footcite:`BEHZADI200790`). However the
#     conclusion was drawn from comparing with an approach that explicitly
#     removes physiological signal, rather than explicitly modelling. Thus
#     compcor might not be a suitable approach for researchers who want
#     explicit description of the source of confounds.
#   - `ica_aroma`: applicable to :term:`fMRIPrep` outputs generated with
#     `--use-aroma`, suffixed with `desc-smoothAROMAnonaggr_bold` only.
#     The regressors contain high pass filtering, white matter, csf, and
#     optionally global signal. The :term:`fMRIPrep` generated
#     `desc-smoothAROMAnonaggr_bold` image requires confounds removal to
#     complete the procedure described in the original approach in
#     :footcite:`Pruim2015`. ICA-AROMA increases the run time of
#     :term:`fMRIPrep`, however, the strategy performs well in various
#     benchmarks (:footcite:`Ciric2017`, :footcite:`Parker2018`).
# The following examples shows how to use the `simple` strategy.

from nilearn.input_data import fmriprep_confounds_strategy

# use default parameters
confounds, sample_mask = fmriprep_confounds_strategy(fmri_filenames,
                                                     denoise_strategy="simple")

# add optional parameter global siganl
confounds, sample_mask = fmriprep_confounds_strategy(fmri_filenames,
                                                     denoise_strategy="simple",
                                                     global_signal="basic")
