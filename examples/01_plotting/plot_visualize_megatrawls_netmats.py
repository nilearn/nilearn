"""
Visualizing Megatrawls Network Matrices from Human Connectome Project
=====================================================================

This example shows how to fetch network matrices data from HCP beta-release
of the Functional Connectivity Megatrawl project.

See :func:`~nilearn.datasets.fetch_megatrawls_netmats` documentation and
the :ref:`dataset description <megatrawls_maps>`
for more details.
"""

# %%
# Fetching the Megatrawls Network matrices
# ----------------------------------------
# Fetching the partial correlation matrices of dimensionality d=300 with
# timeseries method 'eigen regression'
from nilearn.datasets import fetch_megatrawls_netmats

netmats = fetch_megatrawls_netmats(
    dimensionality=300,
    timeseries="eigen_regression",
    matrices="partial_correlation",
)
# Partial correlation matrices array of size (300, 300) are stored in the name
# of 'correlation_matrices'
partial_correlation = netmats.correlation_matrices.to_numpy()

# %%
# Visualization
# -------------
# Import nilearn plotting modules to use its utilities for plotting
# correlation matrices
from nilearn.plotting import plot_matrix, show

title = "Partial correlation matrices\n for d=300"
display = plot_matrix(partial_correlation, title=title)
show()
