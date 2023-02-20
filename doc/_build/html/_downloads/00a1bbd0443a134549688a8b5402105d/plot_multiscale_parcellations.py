"""
Visualizing multiscale functional brain parcellations
=====================================================

This example shows how to download and fetch brain parcellations of
multiple networks using :func:`nilearn.datasets.fetch_atlas_basc_multiscale_2015`
and visualize them using plotting function :func:`nilearn.plotting.plot_roi`.

We show here only three different networks of 'symmetric' version. For more
details about different versions and different networks, please refer to its
documentation.
"""

###############################################################################
# Retrieving multiscale group brain parcellations
# -----------------------------------------------

# import datasets module and use `fetch_atlas_basc_multiscale_2015` function
from nilearn import datasets

parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')

# We show here networks of 64, 197, 444
networks_64 = parcellations['scale064']
networks_197 = parcellations['scale197']
networks_444 = parcellations['scale444']

###############################################################################
# Visualizing brain parcellations
# -------------------------------

# import plotting module and use `plot_roi` function, since the maps are in 3D
from nilearn import plotting

# The coordinates of all plots are selected automatically by itself
# We manually change the colormap of our choice
plotting.plot_roi(networks_64, cmap=plotting.cm.bwr,
                  title='64 regions of brain clusters')

plotting.plot_roi(networks_197, cmap=plotting.cm.bwr,
                  title='197 regions of brain clusters')

plotting.plot_roi(networks_444, cmap=plotting.cm.bwr_r,
                  title='444 regions of brain clusters')

plotting.show()
