"""
Visualizing Megatrawls Network Matrices from Human Connectome Project
=====================================================================

This example shows how to fetch network matrices data from HCP beta-release
of the Functional Connectivity Megatrawl project.

See :func:`nilearn.datasets.fetch_megatrawls_netmats` documentation for more details.
"""


def plot_matrix(matrix, title):
    plt.figure()
    plt.imshow(matrix, interpolation="nearest", cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title(title)

################################################################################
# Fetch the network matrices data of dimensionalities d=100 and d=300 for
# timeseries method 'eigen regression' by importing datasets module
from nilearn import datasets

netmats = datasets.fetch_megatrawls_netmats(dimensionality=[300, 100],
                                            timeseries=['eigen_regression'],
                                            matrices=['partial_correlation'])

# Output matrices are returned according to the sequence of the given inputs.
# Partial correlation matrix arrays: array 1 has matrix with dimensionality=300
# and array 2 has matrix with dimensionality=100
correlation_matrices = netmats.partial_correlation

# Array of given dimensions
dimensions_partial = netmats.dimensions_partial

# Array of timeseries method repeated for total number of given dimensions
timeseries_partial = netmats.timeseries_partial


################################################################################
# Visualization
# Importing matplotlib and nilearn plotting modules to use its utilities for
# plotting correlation matrices
import matplotlib.pyplot as plt
from nilearn import plotting

for matrix, dim, tserie in zip(correlation_matrices, dimensions_partial,
                               timeseries_partial):
    title = 'Partial correlation matrices of d=%d & timeseries=%s' % (dim, tserie)
    plot_matrix(matrix, title)

plt.show()
