"""
Plotting contours of a region of interest on a surface
======================================================
This example showcases how to display contours of regions of interest to a surface using :func:`nilearn.plotting.plot_surf_contours` and how to overlay them on top a statistical surface map from :func:`nilearn.plotting.plot_surf_stat_map`.

"""
##############################################################################
# Get a statistical map and cortical mesh
# ---------------------------------------

from nilearn import datasets

motor_images = datasets.fetch_neurovault_motor_task()
stat_img = motor_images.images[0]

fsaverage = datasets.fetch_surf_fsaverage()

from nilearn import surface
texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)


##############################################################################
# Use an atlas and choose regions to outline
# ------------------------------------------

import numpy as np

destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = destrieux_atlas['map_right']

# these are the regions we want to outline
regions_dict = {b'G_postcentral': 'Postcentral gyrus',
                b'G_precentral': 'Precentral gyrus'}

# get indices in atlas for these labels
regions_indices = [np.where(np.array(destrieux_atlas['labels']) == region)[0][0]
                   for region in regions_dict]

labels = list(regions_dict.values())

##############################################################################
# Display contours on a surface
# -----------------------------

from nilearn import plotting

plotting.plot_surf_contours(fsaverage.infl_right, parcellation, labels=labels,
                            levels=regions_indices, legend=True, colors=['g', 'k'],
                            hemi='right', bg_map=fsaverage.sulc_right,
                            title='Surface right hemisphere')
plotting.show()


##############################################################################
# Display outlines of the regions of interest on top of a statistical map
# -----------------------------------------------------------------------

figure = plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                                     title='Surface right hemisphere', colorbar=True,
                                     threshold=1., bg_map=fsaverage.sulc_right)

plotting.plot_surf_contours(fsaverage.infl_right, parcellation, labels=labels,
                            levels=regions_indices, figure=figure, legend=True,
                            colors=['g', 'k'])
plotting.show()
