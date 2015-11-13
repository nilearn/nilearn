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
# timeseries method ts3
print(" -- Fetching Network Matrices -- ")
netmats = datasets.fetch_megatrawls_netmats(dimensionality=[100, 300],
                                            timeseries='eigen_regression',
                                            matrices=['correlation', 'partial_correlation'])
correlation_matrices_100 = netmats.d100_eigen_regression_correlation
correlation_matrices_300 = netmats.d300_eigen_regression_correlation

partial_correlation_matrices_100 = netmats.d100_eigen_regression_partial_correlation
partial_correlation_matrices_300 = netmats.d300_eigen_regression_partial_correlation

# Visualization
print(" -- Plotting correlation matrices -- ")
correlation_matrices = {
    'Correlation matrices of dimensionality d=100': correlation_matrices_100,
    'Correlation matrices of dimensionality d=300': correlation_matrices_300,
    'Partial correlation matrices of dimensionality d=100': partial_correlation_matrices_100,
    'Partial correlation matrices of dimensionality d=300': partial_correlation_matrices_300
    }

for title, matrices in sorted(correlation_matrices.items()):
    plot_mats(matrices, title)

plt.show()
