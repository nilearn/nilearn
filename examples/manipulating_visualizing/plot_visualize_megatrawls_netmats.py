"""
Visualizing Megatrawls Network Matrices from Human Connectome Project
=====================================================================

This example shows how to visualize network matrices fetched from
HCP beta-release of the Functional Connectivity Megatrawl

For this, we need a fetcher named as :func:`nilearn.datasets.fetch_megatrawls_netmats` in
nilearn.datasets

Please see related documentation for more details.
"""


def plot_mats(netmats, title):
    plt.figure()
    plt.title(title)
    plt.imshow(netmats, interpolation="nearest")
    plt.colorbar()

import numpy as np
from nilearn import datasets
# Fetches the network matrices dimensionalities d=100 and d=300 for
# timeseries method ts3
print(" -- Fetching Network Matrices -- ")
netmats = datasets.fetch_megatrawls_netmats(
    choice_dimensionality=['d100', 'd300'], choice_timeseries='ts3')

# Converting netmats text files to numpy arrays
full_correlation_matrices_d100 = np.genfromtxt(netmats.FullCorrelation[0])
full_correlation_matrices_d300 = np.genfromtxt(netmats.FullCorrelation[1])

partial_correlation_matrices_d100 = np.genfromtxt(netmats.PartialCorrelation[0])
partial_correlation_matrices_d300 = np.genfromtxt(netmats.PartialCorrelation[1])

# Visualization
import matplotlib.pyplot as plt
print(" -- Showing the matrices -- ")
list_ = {
    'Full Correlation matrices of dimensionality d=100': full_correlation_matrices_d100,
    'Full Correlation matrices of dimensionality d=300': full_correlation_matrices_d300,
    'Partial Correlation matrices of dimensionality d=100': partial_correlation_matrices_d100,
    'Partial Correlation matrices of dimensionality d=300': partial_correlation_matrices_d300
    }

for title_, matrices in sorted(list_.items()):
    plot_mats(matrices, title_)

plt.show()
