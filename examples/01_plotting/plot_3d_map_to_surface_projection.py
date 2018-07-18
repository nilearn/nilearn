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

fsaverage = datasets.fetch_surf_fsaverage()

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


##############################################################################
# Plot with higher-resolution mesh
# --------------------------------
#
# `fetch_surf_fsaverage` takes a "mesh" argument which specifies
# wether to fetch the low-resolution fsaverage5 mesh, or the high-resolution
# fsaverage mesh. using mesh="fsaverage" will result in more memory usage and
# computation time, but finer visualizations.

big_fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
big_texture = surface.vol_to_surf(localizer_tmap, big_fsaverage.pial_right)

plotting.plot_surf_stat_map(big_fsaverage.infl_right,
                            big_texture, hemi='right',
                            title='Surface right hemisphere: fine mesh',
                            threshold=1., bg_map=big_fsaverage.sulc_right)


plotting.show()


##############################################################################
# 3D visualization in a web browser
# ---------------------------------
# An alternative to :func:`plot_surf_stat_map` is to use
# :func:`nilearn.plotting.view_surf` or
# :func:`nilearn.plotting.view_img_on_surf` that give more interactive
# visualizations in a web browser. See :ref:`interactive-surface-plotting` for
# more details.
