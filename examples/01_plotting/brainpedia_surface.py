"""
Making a surface plot of a 3D statistical map
=============================================

Download a statistical map from neurovault and a cortical mesh (fsaverage),
project the 3D data on the mesh and display a surface plot.
"""

from matplotlib import pyplot as plt

import nilearn.image
import nilearn.datasets
from nilearn.plotting import surf_plotting

##############################################################################
# Get a statistical map
# ---------------------

brainpedia = nilearn.datasets.fetch_neurovault_ids(image_ids=(32015,))
image = brainpedia.images[0]

##############################################################################
# Get a cortical mesh
# -------------------

fsaverage = nilearn.datasets.fetch_surf_fsaverage5()

##############################################################################
# Sample the 3D data around each node of the mesh
# -----------------------------------------------

texture = surf_plotting.niimg_to_surf_data(image, fsaverage.pial_left, kind='line')


##############################################################################
# Plot the result
# ---------------

surf_plotting.plot_surf_stat_map(fsaverage.infl_left, texture, cmap='bwr')

plt.show()
