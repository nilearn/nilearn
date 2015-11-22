"""
Visualizing Megatrawls Network Matrices from Human Connectome Project
=====================================================================

This example shows how to visualize network matrices fetched from
HCP beta-release of the Functional Connectivity Megatrawl

See :func:`nilearn.datasets.fetch_megatrawls_netmats` documentation for more details.
"""
import matplotlib.pyplot as plt
import numpy as np

from nilearn import datasets
from nilearn import plotting


def plot_mats(netmats, title):
    plt.figure()
    plt.imshow(netmats, interpolation="nearest",
               cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title(title)

# Fetches the network matrices dimensionalities d=100 and d=300 for
# timeseries method multiple regression and eigen regression
print(" -- Fetching Network Matrices -- ")
netmats = datasets.fetch_megatrawls_netmats(dimensionality=[300, 100],
                                            timeseries=['multiple_spatial_regression', 'eigen_regression'],
                                            matrices=['correlation', 'partial_correlation'])

# Visualization
print(" -- Plotting correlation matrices -- ")
for matrices, dim, tseries in zip(
        netmats.correlation, netmats.dimensions_correlation, netmats.timeseries_correlation):
    title = ('Correlation matrices of d=%d & timeseries=%s' % (dim, tseries))
    plot_mats(matrices, title)

print(" -- Plotting partial correlation matrices -- ")
for matrices, dim, tseries in zip(
        netmats.partial_correlation, netmats.dimensions_partial, netmats.timeseries_partial):
    title = ('Partial correlation matrices of d=%d & timeseries=%s' % (dim, tseries))
    plot_mats(matrices, title)

plt.show()
