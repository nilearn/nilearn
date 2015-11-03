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
netmats = datasets.fetch_megatrawls_netmats(
    choice_dimensionality=['d100', 'd300'], choice_timeseries='ts3')

# Converting netmats text files to numpy arrays
full_correlation_matrices_d100 = np.genfromtxt(netmats.Fullcorrelation[0])
full_correlation_matrices_d300 = np.genfromtxt(netmats.Fullcorrelation[1])

partial_correlation_matrices_d100 = np.genfromtxt(netmats.Partialcorrelation[0])
partial_correlation_matrices_d300 = np.genfromtxt(netmats.Partialcorrelation[1])

# Visualization
print(" -- Plotting correlation matrices -- ")
correlation_matrices = {
    'Full correlation matrices of dimensionality d=100': full_correlation_matrices_d100,
    'Full correlation matrices of dimensionality d=300': full_correlation_matrices_d300,
    'Partial correlation matrices of dimensionality d=100': partial_correlation_matrices_d100,
    'Partial correlation matrices of dimensionality d=300': partial_correlation_matrices_d300
    }

for title, matrices in sorted(correlation_matrices.items()):
    plot_mats(matrices, title)

plt.show()
