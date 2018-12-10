"""
Extract signals on spheres from an atlas and plot a connectome
==============================================================

This example shows how to extract signals from spherical regions
centered on coordinates from Power-264 atlas [1] and Dosenbach-160 [2].
We estimate connectome using **sparse inverse covariance**, to recover
the functional brain **networks structure**.

**References**

[1] Power, Jonathan D., et al. "Functional network organization of the
human brain." Neuron 72.4 (2011): 665-678.

[2] Dosenbach N.U., Nardos B., et al. "Prediction of individual brain maturity
using fMRI.", 2010, Science 329, 1358-1361.

"""

###############################################################################
# Load fMRI data and Power atlas
# ------------------------------
#
# We are going to use a single subject from the ADHD dataset.
from nilearn import datasets

adhd = datasets.fetch_adhd(n_subjects=1)

###############################################################################
# We store the paths to its functional image and the confounds file.
fmri_filename = adhd.func[0]
confounds_filename = adhd.confounds[0]
print('Functional image is {0},\nconfounds are {1}.'.format(fmri_filename,
      confounds_filename))

###############################################################################
# We fetch the coordinates of Power atlas.
power = datasets.fetch_coords_power_2011()
print('Power atlas comes with {0}.'.format(power.keys()))

###############################################################################
# Compute within spheres averaged time-series
# -------------------------------------------
#
# We can compute the mean signal within **spheres** of a fixed radius around
# a sequence of (x, y, z) coordinates with the object
# :class:`nilearn.input_data.NiftiSpheresMasker`.
# So we collect the regions coordinates in a numpy array
import numpy as np

coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T

print('Stacked power coordinates in array of shape {0}.'.format(coords.shape))

###############################################################################
# and define spheres masker, with small enough radius to avoid regions overlap.
from nilearn import input_data

spheres_masker = input_data.NiftiSpheresMasker(
    seeds=coords, smoothing_fwhm=4, radius=5.,
    detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=2.5)

###############################################################################
# Voxel-wise time-series within each sphere are averaged. The resulting signal
# is then prepared by the masker object: Detrended, cleaned from counfounds,
# band-pass filtered and **standardized to 1 variance**.
timeseries = spheres_masker.fit_transform(fmri_filename,
                                          confounds=confounds_filename)

###############################################################################
# Estimate correlations
# ---------------------
#
# All starts with the estimation of the signals **covariance** matrix. Here the
# number of ROIs exceeds the number of samples,
print('time series has {0} samples'.format(timeseries.shape[0]))

###############################################################################
# in which situation the graphical lasso **sparse inverse covariance**
# estimator captures well the covariance **structure**.
from sklearn.covariance import GraphLassoCV

covariance_estimator = GraphLassoCV(verbose=1)

###############################################################################
# We just fit our regions signals into the `GraphLassoCV` object
covariance_estimator.fit(timeseries)

###############################################################################
# and get the ROI-to-ROI covariance matrix.
matrix = covariance_estimator.covariance_
print('Covariance matrix has shape {0}.'.format(matrix.shape))

###############################################################################
# Plot matrix and graph
# ---------------------
#
# We use nilearn.plotting.plot_matrix to visualize our correlation matrix
# and display the graph of connections with `nilearn.plotting.plot_connectome`.
from nilearn import plotting

plotting.plot_matrix(matrix, vmin=-1., vmax=1., colorbar=True,
                     title='Power correlation matrix')

# Tweak edge_threshold to keep only the strongest connections.
plotting.plot_connectome(matrix, coords, title='Power correlation graph',
                         edge_threshold='99.8%', node_size=20, colorbar=True)

###############################################################################
# Note the 1. on the matrix diagonal: These are the signals variances, set to
# 1. by the `spheres_masker`. Hence the covariance of the signal is a
# correlation matrix

###############################################################################
# Connectome extracted from Dosenbach's atlas
# -------------------------------------------
#
# We repeat the same steps for Dosenbach's atlas.
dosenbach = datasets.fetch_coords_dosenbach_2010()

coords = np.vstack((
    dosenbach.rois['x'],
    dosenbach.rois['y'],
    dosenbach.rois['z'],
)).T

spheres_masker = input_data.NiftiSpheresMasker(
    seeds=coords, smoothing_fwhm=4, radius=4.5,
    detrend=True, standardize=True, low_pass=0.1, high_pass=0.01, t_r=2.5)

timeseries = spheres_masker.fit_transform(fmri_filename,
                                          confounds=confounds_filename)

covariance_estimator = GraphLassoCV()
covariance_estimator.fit(timeseries)
matrix = covariance_estimator.covariance_

plotting.plot_matrix(matrix, vmin=-1., vmax=1., colorbar=True,
                     title='Dosenbach correlation matrix')

plotting.plot_connectome(matrix, coords, title='Dosenbach correlation graph',
                         edge_threshold="99.7%", node_size=20, colorbar=True)

###############################################################################
# We can easily identify the Dosenbach's networks from the matrix blocks.
print('Dosenbach networks names are {0}'.format(np.unique(dosenbach.networks)))

plotting.show()
