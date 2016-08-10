"""
Extract signals and plot a connectome for atlas-based spheres
=============================================================

This example shows how to extract preprocessed signals from spherical regions
centered on coordinates from Power-264 atlas [1] and Dosenbach-160 [2].
We estimate connectome using sparse inverse covariance, to recover
the functional brain networks structure.

References
----------
[1] Power, Jonathan D., et al. "Functional network organization of the
human brain." Neuron 72.4 (2011): 665-678.

[2] Dosenbach N.U., Nardos B., et al. "Prediction of individual brain maturity
using fMRI.", 2010, Science 329, 1358-1361.

"""

###############################################################################
# Atlases loading
# ---------------

###############################################################################
# We fetch the coordinates of Power atlas
from nilearn import datasets

power = datasets.fetch_coords_power_2011()
print(power.keys())

###############################################################################
# and the coordinates, regions labels and networks names of the Dosenbach
# atlas.
dosenbach = datasets.fetch_coords_dosenbach_2010(ordered_regions=True)
print(dosenbach.keys())

###############################################################################
# Note the use of the parameter *ordered_regions*. It allows to have ROIs
# **ordered** with respect to **functional networks**, which come at number
# of 6.
import numpy as np

print(np.unique(dosenbach.networks))

###############################################################################
# Computing within spheres averaged time-series
# ---------------------------------------------

###############################################################################
# We are going to use a single subject from the ADHD dataset.
adhd = datasets.fetch_adhd(n_subjects=1)


###############################################################################
# We can compute the mean signal within **spheres** of a fixed radius around
# a sequence of 3D coordinates with
# :class:`nilearn.input_data.NiftiSpheresMasker`. 
# So we collect the regions coordinates in numpy arrays
power_coords = np.vstack((
    power.rois['x'],
    power.rois['y'],
    power.rois['z'],
)).T

dosenbach_coords = np.vstack((
    dosenbach.rois['x'],
    dosenbach.rois['y'],
    dosenbach.rois['z'],
)).T

###############################################################################
# and define the spheres maskers, with small enough radius to avoid regions
# overlap.
from nilearn import  input_data

power_masker = input_data.NiftiSpheresMasker(
    seeds=power_coords, smoothing_fwhm=4, radius=5.,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5)

dosenbach_masker = input_data.NiftiSpheresMasker(
    seeds=dosenbach_coords, smoothing_fwhm=4, radius=4.5,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5)

###############################################################################
# Voxel-wise time-series within each sphere are **averaged** and then
# **preprocessed** in an All-In-One fashion.

power_timeseries = power_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

dosenbach_timeseries = dosenbach_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

###############################################################################
# Correlation matrices estimation
# -------------------------------
# All starts with the estimation of the signals **covariance**. Here the number
# of samples
print(power_timeseries.shape[0])

###############################################################################
# is less than the number of ROIs for Power atlas
# (and of the same order for Dosenbach atlas). In these situations,
# graphical lasso **sparse inverse covariance** estimator captures
# well the covariance **structure**.
from sklearn.covariance import GraphLassoCV

###############################################################################
# This estimator can be encompassed into the connectivities estimator object
# :class:`nilearn.connectome.ConnectivityMeasure`. We set the connectivity
# *kind* to correlation, to normalize covariance coefficients. 

from nilearn import connectome

correlation_estimator = connectome.ConnectivityMeasure(
    cov_estimator=GraphLassoCV(verbose=True), kind='correlation')

###############################################################################
# The so-defined **correlations estimator** is designed to compute correlation
# matrices for a group of subjects. We thus fit it to a list containing
# our single subject.

power_corr_matrix = correlation_estimator.fit_transform(
    [power_timeseries])[0]

dosenbach_corr_matrix = correlation_estimator.fit_transform(
    [dosenbach_timeseries])[0]


###############################################################################
# Matrices and graphs plotting
# ----------------------------

###############################################################################
# We use *matplotlib* plotting functions to plot matrices.
import matplotlib.pyplot as plt

for atlas in ['Power', 'Dosenbach']:

    if atlas == 'Power':
        corr_matrix = power_corr_matrix
    else:
        corr_matrix = dosenbach_corr_matrix

    plt.figure()

    # Set diagonal to zero, to emphasize structure
    np.fill_diagonal(corr_matrix, 0)
    vmax = np.max(np.abs(corr_matrix))
    plt.imshow(corr_matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
               interpolation='nearest')
    plt.colorbar()
    plt.title(atlas + ' correlation matrix')

###############################################################################
# Connections graphs can be displayed using
# `nilearn.plotting.plot_connectome`. We tweak the *edge_threshold* to keep
# only the strongest connections.

from nilearn import plotting

for atlas in ['Power', 'Dosenbach']:

    if atlas == 'Power':
        corr_matrix = power_corr_matrix
        coords = power_coords
    else:
        corr_matrix = dosenbach_corr_matrix
        coords = dosenbach_coords

    plotting.plot_connectome(corr_matrix, coords,
                             edge_threshold='99.8%', node_size=20,
                             title=atlas + ' correlation connectome')

plotting.show()
