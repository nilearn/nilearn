"""
Loading and plotting of a cortical surface atlas
=================================================

The Destrieux parcellation (Destrieux et al, 2010) in fsaverage5 space as
distributed with Freesurfer is used as the chosen atlas.

The :func:`nilearn.plotting.plot_surf_roi` function is used
to plot the parcellation on the pial surface.

See :ref:`plotting` for more details.

NOTE: Example needs matplotlib version higher than 1.3.1.

References
----------

Destrieux et al, (2010). Automatic parcellation of human cortical gyri and
sulci using standard anatomical nomenclature. NeuroImage, 53, 1.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.06.010.
"""

###############################################################################
# Data fetcher
# ------------

# Retrieve destrieux parcellation in fsaverage5 space from nilearn
from nilearn import datasets

destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

# The parcellation is already loaded into memory
parcellation = destrieux_atlas['map_left']

# Retrieve fsaverage5 surface dataset for the plotting background. It contains
# the surface template as pial and inflated version and a sulcal depth maps
# which is used for shading
fsaverage = datasets.fetch_surf_fsaverage5()

# The fsaverage dataset contains file names pointing to the file locations
print('Fsaverage5 pial surface of left hemisphere is at: %s' %
      fsaverage['pial_left'])
print('Fsaverage5 inflated surface of left hemisphere is at: %s' %
      fsaverage['infl_left'])
print('Fsaverage5 sulcal depth map of left hemisphere is at: %s' %
      fsaverage['sulc_left'])

###############################################################################
# Visualization
# -------------

# Display Destrieux parcellation on fsaverage5 pial surface using nilearn
from nilearn import plotting

plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=parcellation,
                       hemi='left', view='lateral',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5)

###############################################################################
# Display Destrieux parcellation on inflated fsaverage5 surface
plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=parcellation,
                       hemi='left', view='lateral',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5)

###############################################################################
# Display Destrieux parcellation with different views: posterior
plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=parcellation,
                       hemi='left', view='posterior',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5)

###############################################################################
# Display Destrieux parcellation with different views: ventral
plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=parcellation,
                       hemi='left', view='ventral',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5)
plotting.show()
