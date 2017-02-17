"""
Loading and plotting of a cortical surface atlas
=================================================

The Destrieux parcellation (Destrieux et al, 2010) in fsaverage5 space as
distributed with Freesurfer is used as the chosen atlas.

The :func:`nilearn.plotting.plot_surf_roi` function is used
to plot the parcellation on the pial surface.

See :ref:`plotting` for more details.

References
----------

Destrieux et al, (2010). Automatic parcellation of human cortical gyri and
sulci using standard anatomical nomenclature. NeuroImage, 53, 1.
URL http://dx.doi.org/10.1016/j.neuroimage.2010.06.010.
"""

###############################################################################
from nilearn import plotting
from nilearn import datasets

###############################################################################
# Retrieve destrieux parcellation in fsaverage5 space
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()

# The parcellation is already loaded into memory
parcellation = destrieux_atlas['map_left']

# Retrieve fsaverage5 surface dataset, which contains pial and inflated
# version of the surface template as well as sulcal depth maps
fsaverage = datasets.fetch_surf_fsaverage5()

# The fsaverage dataset contains file names pointing to the file locations
print('Fsaverage5 pial surface of left hemisphere is at: %s' %
      fsaverage['pial_left'])
print('Fsaverage5 inflated surface of left hemisphere is at: %s' %
      fsaverage['infl_left'])
print('Fsaverage5 sulcal depth map of left hemisphere is at: %s' %
      fsaverage['sulc_left'])

###############################################################################
# Display Destrieux parcellation on fsaverage5 pial surface
plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=parcellation,
                       hemi='left', view='lateral',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5, cmap='gist_ncar')

###############################################################################
# Display Destrieux parcellation on inflated fsaverage5 surface
plotting.plot_surf_roi(fsaverage['infl_left'], roi_map=parcellation,
                       hemi='left', view='lateral',
                       bg_map=fsaverage['sulc_left'], bg_on_data=True,
                       darkness=.5, cmap='gist_ncar')

plotting.show()
