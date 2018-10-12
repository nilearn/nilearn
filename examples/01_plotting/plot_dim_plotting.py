"""
Controling the contrast of the background when plotting
=========================================================

The `dim` argument controls the contrast of the background.

*dim* modifies the contrast of this image: dim=0 leaves the image
unchanged, negative values of *dim* enhance it, and positive values
decrease it (dim the background).

This *dim* argument may also be useful for the plot_roi function used to
display ROIs on top of a background image.

The background image used in this example is standard MNI template which
is on by default
"""

#########################################################################
# Retrieve the data: the motor task contrast maps
# -----------------------------------------------

from nilearn import datasets

motor_task_images = datasets.fetch_neurovault_motor_task()
# Contrast map of motor task
motor_tmap_filename = motor_task_images.images[0]

###########################################################################
# Plotting with enhancement of background image (MNI template) with dim=-.5
# --------------------------------------------------------------------------

from nilearn import plotting
plotting.plot_stat_map(motor_tmap_filename,
                       cut_coords=(37, -24, 58),
                       threshold=3, title="dim=-.5",
                       dim=-.5)

########################################################################
# Plotting with no change of contrast in background image with dim=0
# -------------------------------------------------------------------
plotting.plot_stat_map(motor_tmap_filename,
                       cut_coords=(37, -24, 58),
                       threshold=3, title="dim=0",
                       dim=0)

########################################################################
# Plotting with decrease of constrast in background image with dim=.5
# -------------------------------------------------------------------
plotting.plot_stat_map(motor_tmap_filename,
                       cut_coords=(37, -24, 58),
                       threshold=3, title="dim=.5",
                       dim=.5)

########################################################################
# Plotting with more decrease in constrast with dim=1
# ---------------------------------------------------
plotting.plot_stat_map(motor_tmap_filename,
                       cut_coords=(37, -24, 58),
                       threshold=3, title="dim=1",
                       dim=1)

plotting.show()
