"""
Glass brain plotting with contours
===================================

A simple example showing how to plot statistical analysis results using
:func:`nilearn.plotting.plot_glass_brain` with contours.

See Section 4.3. in :ref:`plotting` for more details about display
objects in Nilearn.

See :func:`nilearn.datasets.fetch_localizer_button_task` for details about
the data and its experiments.
"""

###############################################################################
# Downloading datasets from Internet
# -----------------------------------
#
# Nilearn comes with functions that download public data from Internet
#
# Let us first identify where the data is downloaded and stored on our disk:
#
from nilearn import datasets
print('Datasets are stored in: %r' % datasets.get_data_dirs())

###############################################################################
# All datasets that are shipped with Nilearn will be stored at this location

##############################################################################
# Let us now retrieve a motor task contrast maps corresponding to second
# subject from a localizer experiment
tmap_filenames = datasets.fetch_localizer_button_task(n_subjects=[2])['tmaps']
print(tmap_filenames)

##############################################################################
# tmap_filenames is returned as a list. We need to take first one
tmap_filename = tmap_filenames[0]


##############################################################################
# Visualization with contours
# ----------------------------
#
# To visualize statistical images with contours, we call the plotting function
# into variable from which we can use specific display features which are
# inherited automatically. In this case, we focus on using `add_contours`

from nilearn import plotting

# First, we initialize the plotting function into "display" and first
# argument set to None since statistical results must be initialized
# calling with add_contours
display = plotting.plot_glass_brain(None)
# Here, we project statistical maps
display.add_contours(tmap_filename)
# and a title
display.title('"tmap_filename" on glass brain without threshold')

##############################################################################
# Visualizing with specific level (cut-off) in the statistical map

# Here, we set the threshold using parameter called `levels` with value given
# in a list and choosing color to Red.
display = plotting.plot_glass_brain(None)
display.add_contours(tmap_filename, levels=[3.], colors='r')
display.title('"tmap_filename" on glass brain with threshold')

##############################################################################
# Visualizing with black background

# We can set black background using black_bg=True
display = plotting.plot_glass_brain(None, black_bg=True)
display.add_contours(tmap_filename, levels=[3.], colors='g')
display.title('"tmap_filename" on glass brain with black background')
##############################################################################
# Visualizing in both hemispheres left 'l' and right 'r'

# Now, display_mode is chosen as 'lr' for both hemispheric plots
display = plotting.plot_glass_brain(None, display_mode='lr')
display.add_contours(tmap_filename, levels=[3.], colors='r')
display.title('"tmap_filename" on glass brain only "l" "r" hemispheres')
##############################################################################
# Visualizing with positive and negative sign of activations

# By default parameter `plot_abs` is True and sign of activations can done by
# changing `plot_abs` to False
display = plotting.plot_glass_brain(None, plot_abs=False, display_mode='lzry')
display.add_contours(tmap_filename)
display.title("Contours with both sign of activations without threshold")

##############################################################################
# Visualizing both sign of activations with threshold (positive and negative)

# positive threshold implies positive value in a list and negative threshold
# implies negative value (both values are used as cut-off)
display = plotting.plot_glass_brain(None, plot_abs=False)
# positive level and negative level and each level with different colors as a
# list. Additionally, with thick line widths
display.add_contours(tmap_filename, levels=[-2.8, 3.], colors=['b', 'r'],
                     linewidths=4.)
display.title('Contours with both sign of activations with threshold')

# Finally, displaying them
plotting.show()
