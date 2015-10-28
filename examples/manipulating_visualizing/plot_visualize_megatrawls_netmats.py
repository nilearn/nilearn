"""
Visualizing Megatrawls Network Matrices from Human Connectome Project
=====================================================================

This example shows how to visualize network matrices fetched from
HCP beta-release of the Functional Connectivity Megatrawl

For this, we need a fetcher named as `fetch_megatrawls_netmats` in
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
Znet1, Znet2 = datasets.fetch_megatrawls_netmats(
    choice_dimensionality=['d100', 'd300'], choice_timeseries='ts3')

# Converting netmats text files to numpy arrays
netmats1_d100 = np.genfromtxt(Znet1[0])
netmats1_d300 = np.genfromtxt(Znet1[1])

netmats2_d100 = np.genfromtxt(Znet2[0])
netmats2_d300 = np.genfromtxt(Znet2[1])

# Visualization
import matplotlib.pyplot as plt
print(" -- Showing the matrices -- ")
list_ = [netmats1_d100, netmats1_d300, netmats2_d100, netmats2_d300]
titles = ["Full Correlation matrices of dimensionality d=100",
          "Full Correlation matrices of dimensionality d=300",
          "Partial Correlation matrices of dimensionality d=100",
          "Partial Correlation matrices of dimensionality d=300"]
for matrices, title_ in zip(list_, titles):
    plot_mats(matrices, title_)

plt.show()
