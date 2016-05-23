"""
Extracting signals and plotting a connectome for the Power-264 seed-region atlas
================================================================================

This example shows how to extract signals from spherical seed-regions based
on the Power-264 atlas (Power, 2011) and estimating a connectome using sparse
inverse covariance.

Power, Jonathan D., et al. "Functional network organization of the
human brain." Neuron 72.4 (2011): 665-678.

"""

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, connectome, plotting, input_data


###############################################################################
# Atlas and dataset fetching

# Fetch the coordinates of power atlas
power = datasets.fetch_atlas_power_2011()
power_coords = np.vstack((
    power.rois['x'],
    power.rois['y'],
    power.rois['z'],
)).T

# Fetch the first subject of ADHD dataset
adhd = datasets.fetch_adhd(n_subjects=1)


###############################################################################
# Masking: taking the signal in a sphere of radius 5mm around Power coords

masker = input_data.NiftiSpheresMasker(seeds=power_coords,
                                       smoothing_fwhm=4,
                                       radius=5.,
                                       standardize=True,
                                       detrend=True,
                                       low_pass=0.1,
                                       high_pass=0.01,
                                       t_r=2.5)

timeseries = masker.fit_transform(adhd.func[0], confounds=adhd.confounds[0])

###############################################################################
# Extract and plot correlation matrix

# calculate connectivity and plot Power-264 correlation matrix
connectivity = connectome.ConnectivityMeasure(kind='correlation')
corr_matrix = connectivity.fit_transform([timeseries])[0]
np.fill_diagonal(corr_matrix, 0)
plt.imshow(corr_matrix, vmin=-1., vmax=1., cmap='RdBu_r')
plt.colorbar()
plt.title('Power 264 Connectivity')

# Plot the connectome

plotting.plot_connectome(corr_matrix,
                         power_coords,
                         edge_threshold='99.8%',
                         node_size=20)


###############################################################################
# Extract and plot covariance and sparse covariance

# Compute the sparse inverse covariance
from sklearn.covariance import GraphLassoCV

estimator = GraphLassoCV()
estimator.fit(timeseries)

# Display the covariance
plt.figure(figsize=(5, 5))
plt.imshow(estimator.covariance_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
plt.title('Covariance matrix')

# display the corresponding graph
plotting.plot_connectome(estimator.covariance_,
                         power_coords,
                         title='Covariance connectome',
                         edge_threshold='99.8%',
                         node_size=20)

# Display the sparse inverse covariance
plt.figure(figsize=(5, 5))
plt.imshow(estimator.precision_, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.RdBu_r)
plt.title('Precision matrix')

# And now display the corresponding graph
plotting.plot_connectome(estimator.precision_, power_coords,
                         title='Precision connectome',
                         edge_threshold="99.8%",
                         node_size=20)
plotting.show()
