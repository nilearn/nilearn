"""
Making a surface plot of a 3D statistical map
=============================================

project a 3D statistical map onto a cortical mesh and display a surface plot.

"""

from nilearn import datasets
from nilearn import plotting

##############################################################################
# Get a statistical map
# ---------------------

localizer_dataset = datasets.fetch_localizer_button_task()
localizer_tmap = localizer_dataset.tmaps[0]

##############################################################################
# Get a cortical mesh
# -------------------

fsaverage = datasets.fetch_surf_fsaverage5()

##############################################################################
# Sample the 3D data around each node of the mesh
# -----------------------------------------------

texture = plotting.niimg_to_surf_data(localizer_tmap, fsaverage.pial_right)

##############################################################################
# Plot the result
# ---------------

plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere')

plotting.plot_glass_brain(localizer_tmap, display_mode='r', plot_abs=False,
                          title='Glass brain')

plotting.show()
