"""
More plotting tools from nilearn
================================

In this example, we demonstrate how to use plotting options from
nilearn essential in visualizing brain image analysis results.

We emphasize the use of parameters such as `display_mode` and `cut_coords`
with plotting function :func:`nilearn.plotting.plot_stat_map`. Also,
we show how to use various features such as `add_edges`, `add_contours`,
`add_markers` essential in visualizing regions of interest images or
mask images overlaying on subject specific anatomical/EPI image.
The display features shown here are inherited from the
:class:`nilearn.plotting.displays.OrthoSlicer` class.

The parameter `display_mode` is used to draw brain slices along given
specific directions, where directions can be one of 'ortho',
'x', 'y', 'z', 'xy', 'xz', 'yz'. whereas parameter `cut_coords`
is used to specify a limited number of slices to visualize along given
specific slice direction. The parameter `cut_coords` can also be used
to draw the specific cuts in the slices by giving its particular
coordinates in MNI space accordingly with particular slice direction.
This helps us point to the activation specific location of the brain slices.

See :ref:`plotting` for more details.
"""

###############################################################################
# First, we retrieve data from nilearn provided (general-purpose) datasets

from nilearn import datasets

# haxby dataset to have anatomical image, EPI images and masks
haxby_dataset = datasets.fetch_haxby(n_subjects=1)
haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

# localizer dataset to have contrast maps
localizer_dataset = datasets.fetch_localizer_button_task(get_anats=True)
localizer_anat_filename = localizer_dataset.anats[0]
localizer_tmap_filename = localizer_dataset.tmaps[0]

########################################
# Now, we show from here how to visualize the retrieved datasets using plotting
# tools from nilearn.

from nilearn import plotting

########################################
# Visualizing contrast map in three different orthogonal views - 'sagittal',
# 'coronal' and 'axial' with coordinate positions in each view are given
# of interest manually also with colorbar on the right side of the plots.

# The first argument is a path to the filename of a constrast map,
# optional argument `display_mode` is given as string 'ortho' to visualize
# the map in three specific directions xyz and the optional `cut_coords`
# argument, is here a list of integers denotes coordinates of each slice
# in the order [x, y, z]. By default the `colorbar` argument is set to True
# in plot_stat_map.
plotting.plot_stat_map(localizer_tmap_filename, display_mode='ortho',
                       cut_coords=[36, -27, 60],
                       title="display_mode='ortho', cut_coords=[36, -27, 60]")

########################################
# Visualizing contrast map in single view 'axial' with maximum number of
# slices in this view are limited to 5. The coordinates to cut the slices
# are selected automatically.

# In this type of visualization, the `display_mode` argument is given as
# string 'z' for axial direction and `cut_coords` as integer 5 without a
# list implies that number of cuts in the slices should be maximum of 5.
plotting.plot_stat_map(localizer_tmap_filename, display_mode='z', cut_coords=5,
                       title="display_mode='z', cut_coords=5")

########################################
# Visualizing contrast map in another single view 'sagittal' and also showing
# how to select two slices of particular interest manually by giving the
# coordinates to cut each slice.

# In this type, `display_mode` should be given as string 'x' for sagittal
# view and coordinates should be given as integers in a list
plotting.plot_stat_map(localizer_tmap_filename, display_mode='x',
                       cut_coords=[-36, 36],
                       title="display_mode='x', cut_coords=[-36, 36]")

########################################
# Now constrast map is visualized in 'coronal' view with single cut where
# coordinates are located automatically

# For coronal view, `display_mode` is given as string 'y' and `cut_coords`
# as integer 1 not as a list for single cut
plotting.plot_stat_map(localizer_tmap_filename, display_mode='y', cut_coords=1,
                       title="display_mode='y', cut_coords=1")

########################################
# Now contrast map is shown without a colorbar on the right side.

# The argument `colorbar` should be given as False to show plots without
# a colorbar on the right side.
plotting.plot_stat_map(localizer_tmap_filename, display_mode='z',
                       cut_coords=1, colorbar=False,
                       title="display_mode='z', cut_coords=1, colorbar=False")

########################################
# Now we visualize the contrast map with two views - 'sagittal' and 'axial'
# and coordinates are given manually to select particular cuts in two views.

# argument display_mode='xz' where 'x' for sagittal and 'z' for axial view.
# argument `cut_coords` should match with input number of views therefore two
# integers should be given in a list to select the slices to be displayed
plotting.plot_stat_map(localizer_tmap_filename, display_mode='xz',
                       cut_coords=[36, 60],
                       title="display_mode='xz', cut_coords=[36, 60]")

########################################
# Visualizing the contrast map with 'coronal', 'sagittal' views and coordinates
# are given manually in two cuts with two views.

# display_mode='yx' for coronal and saggital view and coordinates will be
# assigned in the order of direction as [x, y, z]
plotting.plot_stat_map(localizer_tmap_filename, display_mode='yx',
                       cut_coords=[-27, 36],
                       title="display_mode='yx', cut_coords=[-27, 36]")

########################################
# Visualizing contrast map with 'coronal' and 'axial' views with manual
# positioning of coordinates with each directional view

plotting.plot_stat_map(localizer_tmap_filename, display_mode='yz',
                       cut_coords=[-27, 60],
                       title="display_mode='yz', cut_coords=[-27, 60]")

###############################################################################
# In second part, we switch to demonstrating various features add_* from
# nilearn where each specific feature will be helpful in projecting brain
# imaging results for further interpretation.

# Import image processing tool for basic processing of functional brain image
from nilearn import image

# Compute voxel-wise mean functional image across time dimension. Now we have
# functional image in 3D assigned in mean_haxby_img
mean_haxby_img = image.mean_img(haxby_func_filename)

########################################
# Now let us see how to use `add_edges`, method useful for checking
# coregistration by overlaying anatomical image as edges (red) on top of
# mean functional image (background), both being of same subject.

# First, we call the `plot_anat` plotting function, with a background image
# as first argument, in this case the mean fMRI image.
display = plotting.plot_anat(mean_haxby_img, title="add_edges")

# We are now able to use add_edges method inherited in plotting object named as
# display. First argument - anatomical image  and by default edges will be
# displayed as red 'r', to choose different colors green 'g' and  blue 'b'.
display.add_edges(haxby_anat_filename)

########################################
# Note that the edges of a non-coregistered anatomical image may partially
# match a mean functional image, because plotting is done in real world space.
# So one has to look at fine details.

########################################
# Plotting outline of the mask (red) on top of the mean EPI image with
# `add_contours`. This method is useful for region specific interpretation
# of brain images

# As seen before, we call the `plot_anat` function with a background image
# as first argument, in this case again the mean fMRI image and argument
# `cut_coords` as list for manual cut with coordinates pointing at masked
# brain regions
display = plotting.plot_anat(mean_haxby_img, title="add_contours",
                             cut_coords=[28, -34, -22])
# Now use `add_contours` in display object with the path to a mask image from
# the Haxby dataset as first argument and argument `levels` given as list
# of values to select particular level in the contour to display and argument
# `colors` specified as red 'r' to see edges as red in color.
# See help on matplotlib.pyplot.contour to use more options with this method
display.add_contours(haxby_mask_filename, levels=[0.5], colors='r')

########################################
# Plotting outline of the mask (blue) with color fillings using same method
# `add_contours`.

display = plotting.plot_anat(mean_haxby_img,
                             title="add_contours with filled=True",
                             cut_coords=[28, -34, -22])

# By default, no color fillings will be shown using `add_contours`. To see
# contours with color fillings use argument filled=True. contour colors are
# changed to blue 'b' with alpha=0.7 sets the transparency of color fillings.
# See help on matplotlib.pyplot.contourf to use more options given that filled
# should be True
display.add_contours(haxby_mask_filename, filled=True, alpha=0.7,
                     levels=[0.5], colors='b')

#########################################
# Plotting seed regions of interest as spheres using new feature `add_markers`
# with MNI coordinates of interest.

display = plotting.plot_anat(mean_haxby_img, title="add_markers",
                             cut_coords=[28, -34, -22])

# Coordinates of seed regions should be specified in first argument and second
# argument `marker_color` denotes color of the sphere in this case yellow 'y'
# and third argument `marker_size` denotes size of the sphere
coords = [(28, -34, -22)]
display.add_markers(coords, marker_color='y', marker_size=100)

###############################################################################
# Finally, saving the plots to file with two different ways

# Contrast maps plotted with function `plot_stat_map` can be saved using an
# inbuilt parameter output_file as filename + .extension as string. Valid
# extensions are .png, .pdf, .svg
plotting.plot_stat_map(localizer_tmap_filename,
                       title='Using plot_stat_map output_file',
                       output_file='plot_stat_map.png')

########################################
# Another way of saving plots is using 'savefig' option from display object
display = plotting.plot_stat_map(localizer_tmap_filename,
                                 title='Using display savefig')
display.savefig('plot_stat_map_from_display.png')
# In non-interactive settings make sure you close your displays
display.close()

plotting.show()
