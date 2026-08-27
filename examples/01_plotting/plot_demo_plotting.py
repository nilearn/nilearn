# %%
"""
Plotting tools in nilearn
=========================

Nilearn comes with a set of plotting functions for easy visualization of
Nifti-like images such as statistical maps mapped onto anatomical images
or onto glass brain representation, anatomical images, functional/EPI images,
region specific mask images.

See :ref:`plotting` for more details.
"""

# %%
# We will first retrieve data from nilearn provided (general-purpose)
# datasets.

from nilearn.datasets import fetch_haxby, load_sample_motor_activation_image

# Fetch the Haxby dataset to have EPI images and masks.
haxby_dataset = fetch_haxby()

# print basic information on the dataset
print(
    f"First subject anatomical nifti image (3D) is at: {haxby_dataset.anat[0]}"
)
print(
    f"First subject functional nifti image (4D) is at: {haxby_dataset.func[0]}"
)

haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

# one motor activation map
stat_img = load_sample_motor_activation_image()


# %%
# Nilearn plotting functions
# --------------------------
#
# Plotting statistical maps: `plot_stat_map`
# ``````````````````````````````````````````

from nilearn.plotting import plot_stat_map, show

# %%
# Visualizing t-map image on EPI template with manual
# positioning of coordinates using cut_coords given as a list.
plot_stat_map(
    stat_img, threshold=3, title="plot_stat_map", cut_coords=[36, -27, 66]
)

# %%
# It's also possible to visualize volumes in a LR-flipped "radiological" view
# by setting radiological=True:
plot_stat_map(
    stat_img,
    threshold=3,
    title="plot_stat_map",
    cut_coords=[36, -27, 66],
    radiological=True,
)

# %%
# Calling ``show`` function
# from :mod:`nilearn.plotting` package is necessary to display the figure
# when running as a script outside IPython.

show()


# %%
# Making interactive visualizations: `view_img`
# `````````````````````````````````````````````
# An alternative to :func:`~nilearn.plotting.plot_stat_map` is to use
# :func:`~nilearn.plotting.view_img` that gives more interactive
# visualizations in a web browser.
# See :ref:`interactive-stat-map-plotting` for more details.
from nilearn.plotting import view_img

view = view_img(stat_img, threshold=3)
# In a notebook, if ``view`` is the output of a cell, it will
# be displayed below the cell.
view

# %%
# Uncomment this to open the plot in a web browser:

# view.open_in_browser()

# %%
# It's also possible to visualize volumes in a LR-flipped "radiological" view
# by setting radiological=True:
view_radio = view_img(
    stat_img, threshold=3, title="radiological view", radiological=True
)
view_radio

# %%
# Uncomment this to open the plot in a web browser:

# view_radio.open_in_browser()

# %%
# Plotting statistical maps in a glass brain: `plot_glass_brain`
# ``````````````````````````````````````````````````````````````
#
# Now, the t-map image is mapped on glass brain representation where glass
# brain is always a fixed background template.
from nilearn.plotting import plot_glass_brain

plot_glass_brain(stat_img, title="plot_glass_brain", threshold=3)

show()

# %%
# Plotting anatomical images: `plot_anat`
# ```````````````````````````````````````
#
# Visualizing anatomical image of the Haxby dataset
from nilearn.plotting import plot_anat

plot_anat(haxby_anat_filename, title="plot_anat")

show()

# %%
# Plotting ROIs (here the mask): `plot_roi`
# `````````````````````````````````````````
#
# Visualizing ventral temporal region image from the Haxby dataset overlaid on
# subject specific anatomical image
# with coordinates positioned automatically on region of interest (roi).
from nilearn.plotting import plot_roi

plot_roi(haxby_mask_filename, bg_img=haxby_anat_filename, title="plot_roi")

show()

# %%
# Plotting :term:`EPI` image: `plot_epi`
# ``````````````````````````````````````


# Compute the voxel_wise mean of functional images across time,
# basically reducing the functional image from 4D to 3D.
from nilearn.image import mean_img
from nilearn.plotting import plot_epi

mean_haxby_img = mean_img(haxby_func_filename)

# Visualizing mean image (3D)
plot_epi(mean_haxby_img, title="plot_epi")

show()

# %%
# Thresholding plots
# ------------------
#
# Using ``threshold`` value alongside with ``vmin`` and ``vmax`` parameters
# enable us to mask certain values in the image.
#
# Plotting without threshold
# ``````````````````````````

plot_stat_map(
    stat_img,
    display_mode="ortho",
    cut_coords=[36, -27, 60],
    title="No plotting threshold",
)

show()

# %%
# Plotting threshold set to 1
# ```````````````````````````
#
# When plotting threshold is set to 1, the values between -1 and 1
# are masked in the plot.

plot_stat_map(
    stat_img,
    threshold=1,
    display_mode="ortho",
    cut_coords=[36, -27, 60],
    title="plotting threshold=1",
)

show()

# %%
# Plotting threshold set to 1 with ``vmin=0``
# ```````````````````````````````````````````
#
# Setting ``vmin=0``, it is possible to plot only positive image values.

plot_stat_map(
    stat_img,
    threshold=1,
    cmap="inferno",
    display_mode="ortho",
    cut_coords=[36, -27, 60],
    title="plotting threshold=1, vmin=0",
    vmin=0,
)

show()

# %%
# Plotting threshold set to 1 with ``vmax=0``
# ```````````````````````````````````````````
#
# Setting ``vmax=0``, it is possible to plot only negative image values.

plot_stat_map(
    stat_img,
    threshold=1,
    cmap="inferno",
    display_mode="ortho",
    cut_coords=[36, -27, 60],
    title="plotting threshold=1, vmax=0",
    vmax=0,
)

show()

# %%
# Visualizing without a colorbar on the right side
# ------------------------------------------------
#
# The argument ``colorbar`` should be set to ``False`` to show plots
# without a colorbar on the right side.

plot_stat_map(
    stat_img,
    display_mode="ortho",
    cut_coords=[36, -27, 60],
    colorbar=False,
    title="no colorbar",
)

show()

# sphinx_gallery_dummy_images=11
