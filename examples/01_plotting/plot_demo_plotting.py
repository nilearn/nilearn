"""
Plotting tools in nilearn
=========================

Nilearn comes with a set of plotting functions for easy visualization of
Nifti-like images such as statistical maps mapped onto anatomical images
or onto glass brain representation, anatomical images, functional/EPI images,
region specific mask images.

See :ref:`plotting` for more details.
"""

###############################################################################
# Retrieve data from nilearn provided (general-purpose) datasets
# --------------------------------------------------------------

from nilearn import datasets

# haxby dataset to have EPI images and masks
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print("First subject anatomical nifti image (3D) is "
      f"at: {haxby_dataset.anat[0]}")
print(
    "First subject functional nifti image (4D) is "
    f"at: {haxby_dataset.func[0]}"
)

haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

# one motor activation map
stat_img = datasets.load_sample_motor_activation_image()


###############################################################################
# Plotting statistical maps with function `plot_stat_map`
# -------------------------------------------------------

from nilearn import plotting

# Visualizing t-map image on EPI template with manual
# positioning of coordinates using cut_coords given as a list
plotting.plot_stat_map(stat_img,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[36, -27, 66])

###############################################################################
# Making interactive visualizations with function `view_img`
# ----------------------------------------------------------
# An alternative to :func:`nilearn.plotting.plot_stat_map` is to use
# :func:`nilearn.plotting.view_img` that gives more interactive
# visualizations in a web browser. See :ref:`interactive-stat-map-plotting`
# for more details.

view = plotting.view_img(stat_img, threshold=3)
# In a Jupyter notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell
view

##############################################################################

# uncomment this to open the plot in a web browser:
# view.open_in_browser()


###############################################################################
# Plotting statistical maps in a glass brain with function `plot_glass_brain`
# ---------------------------------------------------------------------------
#
# Now, the t-map image is mapped on glass brain representation where glass
# brain is always a fixed background template
plotting.plot_glass_brain(stat_img, title='plot_glass_brain',
                          threshold=3)

###############################################################################
# Plotting anatomical images with function `plot_anat`
# ----------------------------------------------------
#
# Visualizing anatomical image of haxby dataset
plotting.plot_anat(haxby_anat_filename, title="plot_anat")

###############################################################################
# Plotting ROIs (here the mask) with function `plot_roi`
# ------------------------------------------------------
#
# Visualizing ventral temporal region image from haxby dataset overlaid on
# subject specific anatomical image
# with coordinates positioned automatically on region of interest (roi)
plotting.plot_roi(haxby_mask_filename, bg_img=haxby_anat_filename,
                  title="plot_roi")

###############################################################################
# Plotting EPI image with function `plot_epi`
# -------------------------------------------

# Import image processing tool
from nilearn import image

# Compute the voxel_wise mean of functional images across time.
# Basically reducing the functional image from 4D to 3D
mean_haxby_img = image.mean_img(haxby_func_filename)

# Visualizing mean image (3D)
plotting.plot_epi(mean_haxby_img, title="plot_epi")

###############################################################################
# A call to plotting.show is needed to display the plots when running
# in script mode (ie outside IPython)
plotting.show()
