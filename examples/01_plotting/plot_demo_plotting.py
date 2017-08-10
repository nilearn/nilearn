"""
Plotting tools in nilearn
==========================

Nilearn comes with a set of plotting functions for easy visualization of
Nifti-like images such as statistical maps mapped onto anatomical images
or onto glass brain representation, anatomical images, functional/EPI images,
region specific mask images.

See :ref:`plotting` for more details.
"""

###############################################################################
# Retrieve data from nilearn provided (general-purpose) datasets
# ---------------------------------------------------------------

from nilearn import datasets

# haxby dataset to have EPI images and masks
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print('First subject anatomical nifti image (3D) is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is at: %s' %
      haxby_dataset.func[0])  # 4D data

haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

# localizer dataset to have contrast maps
localizer_dataset = datasets.fetch_localizer_button_task(get_anats=True)
localizer_anat_filename = localizer_dataset.anats[0]
localizer_tmap_filename = localizer_dataset.tmaps[0]

###############################################################################
# Plotting statistical maps with function `plot_stat_map`
# --------------------------------------------------------

from nilearn import plotting

# Visualizing t-map image on subject specific anatomical image with manual
# positioning of coordinates using cut_coords given as a list
plotting.plot_stat_map(localizer_tmap_filename, bg_img=localizer_anat_filename,
                       threshold=3, title="plot_stat_map",
                       cut_coords=[36, -27, 66])

###############################################################################
# Plotting statistical maps in a glass brain with function `plot_glass_brain`
# ---------------------------------------------------------------------------
#
# Now, the t-map image is mapped on glass brain representation where glass
# brain is always a fixed background template
plotting.plot_glass_brain(localizer_tmap_filename, title='plot_glass_brain',
                          threshold=3)

###############################################################################
# Plotting anatomical images with function `plot_anat`
# -----------------------------------------------------
#
# Visualizing anatomical image of haxby dataset
plotting.plot_anat(haxby_anat_filename, title="plot_anat")

###############################################################################
# Plotting ROIs (here the mask) with function `plot_roi`
# -------------------------------------------------------
#
# Visualizing ventral temporal region image from haxby dataset overlayed on
# subject specific anatomical image with coordinates positioned automatically on
# region of interest (roi)
plotting.plot_roi(haxby_mask_filename, bg_img=haxby_anat_filename,
                  title="plot_roi")

###############################################################################
# Plotting EPI image with function `plot_epi`
# ---------------------------------------------

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
