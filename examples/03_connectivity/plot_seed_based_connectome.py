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
    seeds=dosenbach_coords, smoothing_fwhm=4, radius=4.,
    standardize=True, detrend=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5)

power_timeseries = power_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

dosenbach_timeseries = dosenbach_masker.fit_transform(
    adhd.func[0], confounds=adhd.confounds[0])

###############################################################################
# Extract and plot correlation matrix

for atlas in ['Power', 'Dosenbach']:

    if atlas == 'Power':
        timeseries = power_timeseries
        coords = power_coords
    else:
        timeseries = dosenbach_timeseries
        coords = dosenbach_coords

    connectivity = connectome.ConnectivityMeasure(kind='correlation')
    corr_matrix = connectivity.fit_transform([timeseries])[0]
    np.fill_diagonal(corr_matrix, 0)

    plt.figure()
    vmax = np.max(np.abs(corr_matrix))
    plt.imshow(corr_matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
               interpolation='nearest')
    plt.colorbar()
    plt.title(atlas + 'correlation matrix')

    # Plot the connectome
    plotting.plot_connectome(corr_matrix, coords,
                             edge_threshold='99.8%', node_size=20,
                             title=atlas + 'correlation connectome')

plotting.show()
