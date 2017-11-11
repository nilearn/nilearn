"""
Making a surface plot of a 3D statistical map
=============================================

Download a statistical map from Neurovault and a cortical mesh (fsaverage),
project the 3D data on the mesh and display a surface plot.

"""

from nilearn import datasets
from nilearn import plotting

##############################################################################
# Get a statistical map
# ---------------------

brainpedia = datasets.fetch_neurovault_ids(image_ids=(32015,))
image = brainpedia.images[0]

##############################################################################
# Get a cortical mesh
# -------------------

fsaverage = datasets.fetch_surf_fsaverage5()

##############################################################################
# Sample the 3D data around each node of the mesh
# -----------------------------------------------

texture = plotting.niimg_to_surf_data(image, fsaverage.pial_left)

##############################################################################
# Plot the result
# ---------------

plotting.plot_surf_stat_map(fsaverage.infl_left, texture, cmap='bwr')

plotting.show()
