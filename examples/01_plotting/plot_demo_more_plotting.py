"""
More plotting tools from nilearn
================================

In this example, we demonstrate how to display results benefitting from
more plotting options available in nilearn.

All options demonstrated here are from existing plotting functions
:func:nilearn.plotting.img_plotting. Particularly, we focus how to use
different display modes `display_mode` to display results in specific
slice directions and also positioning of coordinates on the slices and
number of slices to display using `cut_coords`.

We also demonstrate using various plotting objects from
:class:nilearn.plotting.displays.OrthoSlicer consists of different types
of display methods `add_overlay`, `add_contours`, `add_markers` for
two step visualization of 3D maps.

See :ref:`plotting` for more details.
"""

###############################################################################
# Retrieve the data from nilearn: haxby dataset to have EPI images and masks,
# and localizer dataset to have contrast maps

# Import datasets module
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
# Plotting statistical maps in three slice directions and also manually
# positioning the location of the coordinates on each slice direction

# Import plotting module for visualization
from nilearn import plotting

# Visualizing with default option display_mode='ortho' and coordinates given as
# a list. display_mode='ortho' implies visualizing in three slice directions.
plotting.plot_stat_map(localizer_cmap_filename, display_mode='ortho',
                       cut_coords=[36, -27, 60],
                       title="display_mode='ortho', cut_coords=[36, -27, 60]")

########################################
# Now, plotting results in single direction 'z' with total number of slices to
# be displayed=5

# This visualization requires display_mode given as string 'z' and cut_coords as
# integer.
# Note: the difference in cut_coords as integer and as list.
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

# Import image processing tool
from nilearn import image

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
