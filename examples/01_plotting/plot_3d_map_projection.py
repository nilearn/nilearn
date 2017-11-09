"""
Making a surface plot of a 3D statistical map
=============================================

Download a statistical map from Neurovault and a cortical mesh (fsaverage),
project the 3D data on the mesh and display a surface plot.

"""

import nilearn.image
import nilearn.datasets
import nilearn.plotting

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

texture = nilearn.plotting.niimg_to_surf_data(image, fsaverage.pial_left)


##############################################################################
# Plot the result
# ---------------

nilearn.plotting.plot_surf_stat_map(fsaverage.infl_left, texture, cmap='bwr')

nilearn.plotting.show()
