"""
Extract signals and plot a connectome for seed-based atlases
============================================================

This example shows how to extract signals from spherical seed-based
atlases such as Power-264 atlas (Power, 2011) and Dosenbach-160 (Dosenbach,
2010). We estimate connectome using sparse inverse covariance.

Dosenbach N.U., Nardos B., et al. "Prediction of individual brain maturity
using fMRI.", 2010, Science 329, 1358-1361.

Power, Jonathan D., et al. "Functional network organization of the
human brain." Neuron 72.4 (2011): 665-678.
"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, connectome, plotting, input_data


###############################################################################
# Atlas and dataset fetching

# Fetch the coordinates of power atlas
power = datasets.fetch_coords_power_2011()
power_coords = np.vstack((
    power.rois['x'],
    power.rois['y'],
    power.rois['z'],
)).T

dosenbach = datasets.fetch_coords_dosenbach_2010()
dosenbach_coords = np.vstack((
    dosenbach.rois['x'],
    dosenbach.rois['y'],
    dosenbach.rois['z'],
)).T

# Fetch the first subject of ADHD dataset
adhd = datasets.fetch_adhd(n_subjects=1)


###############################################################################
# Masking: taking the signal in a sphere of radius 5 around Power coords

power_masker = input_data.NiftiSpheresMasker(
    seeds=power_coords, smoothing_fwhm=4, radius=5.,
    standardize=True, detrend=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5)

dosenbach_masker = input_data.NiftiSpheresMasker(
    seeds=dosenbach_coords, smoothing_fwhm=4, radius=5.,
    standardize=True, detrend=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5)

power_timeseries = power_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

dosenbach_timeseries = dosenbach_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

###############################################################################
# Extract and plot correlation matrix

connectivity = connectome.ConnectivityMeasure(kind='correlation')
corr_matrix = connectivity.fit_transform([power_timeseries])[0]
plt.imshow(corr_matrix, vmin=-1., vmax=1., cmap='RdBu_r')
plt.colorbar()
plt.title('Power correlation matrix')

# Plot the connectome
plotting.plot_connectome(corr_matrix,
                         power_coords,
                         edge_threshold='99.8%',
                         node_size=20,
                         title="Power correlation connectome")


###############################################################################
# Extract and plot covariance and sparse covariance

# Compute the sparse inverse covariance
from sklearn.covariance import GraphLassoCV

connectivity = connectome.ConnectivityMeasure(kind='partial_correlation',
                                              estimator=GraphLassoCV())
prec_matrix = connectivity.fit_transform([power_timeseries])[0]

# Display the sparse inverse covariance
plt.imshow(connectivity.precision_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
plt.title('Power sparse partial correlation matrix')

# display the corresponding graph
plotting.plot_connectome(connectivity.precision_,
                         power_coords,
                         title='Power sparse partial correlation connectome',
                         edge_threshold='99.8%',
                         node_size=20)

plotting.show()
