"""
More nilearn plotting
=====================

See :ref:`plotting` for more details.
"""

# The imports from nilearn plotting and image processing
from nilearn import plotting, image

###############################################################################
# Retrieve the data: haxby dataset to have EPI images and masks, and
# localizer dataset to have contrast maps

from nilearn import datasets
haxby_dataset = datasets.fetch_haxby(n_subjects=1)
haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=2,
    get_anats=True)
localizer_anat_filename = localizer_dataset.anats[1]
localizer_cmap_filename = localizer_dataset.cmaps[1]

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='ortho',
                       cut_coords=(36, -27, 60),
                       title="display_mode='ortho', cut_coords=(36, -27, 60)")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='z', cut_coords=5,
                       title="display_mode='z', cut_coords=5")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='x',
                       cut_coords=(-36, 36),
                       title="display_mode='x', cut_coords=(-36, 36)")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='y', cut_coords=1,
                       title="display_mode='x', cut_coords=(-36, 36)")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='z',
                       cut_coords=1, colorbar=False,
                       title="display_mode='z', cut_coords=1, colorbar=False")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='xz',
                       cut_coords=(36, 60),
                       title="display_mode='xz', cut_coords=(36, 60)")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='yx',
                       cut_coords=(-27, 36),
                       title="display_mode='yx', cut_coords=(-27, 36)")

########################################
plotting.plot_stat_map(localizer_cmap_filename, display_mode='yz',
                       cut_coords=(-27, 60),
                       title="display_mode='yz', cut_coords=(-27, 60)")

###############################################################################
# demo display objects with add_* methods
mean_haxby_img = image.mean_img(haxby_func_filename)

# Plot T1 outline on top of the mean EPI (useful for checking coregistration)
display = plotting.plot_anat(mean_haxby_img, title="add_edges")
display.add_edges(haxby_anat_filename)

########################################
# Plotting outline of the mask on top of the EPI
display = plotting.plot_anat(mean_haxby_img, title="add_contours",
                             cut_coords=(28, -34, -22))
display.add_contours(haxby_mask_filename, levels=[0.5], colors='r')

###############################################################################
# demo saving plots to file

plotting.plot_stat_map(localizer_cmap_filename,
                       title='Using plot_stat_map output_file',
                       output_file='plot_stat_map.png')

########################################
display = plotting.plot_stat_map(localizer_cmap_filename,
                                 title='Using display savefig')
display.savefig('plot_stat_map_from_display.png')
# In non-interactive settings make sure you close your displays
display.close()

plotting.show()
