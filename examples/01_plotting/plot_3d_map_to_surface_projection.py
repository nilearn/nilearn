"""
Making a surface plot of a 3D statistical map
=============================================

project a 3D statistical map onto a cortical mesh using
:func:`nilearn.surface.vol_to_surf`. Display a surface plot of the projected
map using :func:`nilearn.plotting.plot_surf_stat_map`.

NOTE: Example needs matplotlib version higher than 1.3.1.

"""

##############################################################################
# Get a statistical map
# ---------------------

from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_button_task()
localizer_tmap = localizer_dataset.tmaps[0]

##############################################################################
# Get a cortical mesh
# -------------------

fsaverage = datasets.fetch_surf_fsaverage5()

##############################################################################
# Sample the 3D data around each node of the mesh
# -----------------------------------------------

from nilearn import surface

texture = surface.vol_to_surf(localizer_tmap, fsaverage.pial_right)

##############################################################################
# Plot the result
# ---------------

from nilearn import plotting

plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere',
                            threshold=1., bg_map=fsaverage.sulc_right)

##############################################################################
# Plot 3D image for comparison
# ----------------------------

plotting.plot_glass_brain(localizer_tmap, display_mode='r', plot_abs=False,
                          title='Glass brain', threshold=2.)

plotting.plot_stat_map(localizer_tmap, display_mode='x', threshold=1.,
                       cut_coords=range(0, 51, 10), title='Slices')

plotting.show()
