"""
Glass brain plotting in nilearn (all options)
=============================================

This example goes through different options of the :func:`nilearn.plotting.plot_glass_brain` function
(including plotting negative values).

See :ref:`plotting` for more plotting functionalities.
"""


###############################################################################
# Retrieve the data
from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_button_task()
localizer_tmap_filename = localizer_dataset.tmaps[0]

###############################################################################
# Demo glass brain plotting.
from nilearn import plotting

# Whole brain sagittal cuts
plotting.plot_glass_brain(localizer_tmap_filename, threshold=3)


###############################################################################
# With a colorbar
plotting.plot_glass_brain(localizer_tmap_filename, threshold=3, colorbar=True)


###############################################################################
# Black background, and only the (x, z) cuts
plotting.plot_glass_brain(localizer_tmap_filename, title='plot_glass_brain',
                          black_bg=True, display_mode='xz', threshold=3)


###############################################################################
# Plotting the sign of the activation
plotting.plot_glass_brain(localizer_tmap_filename, threshold=0, colorbar=True,
                          plot_abs=False)


###############################################################################
# The sign of the activation and a colorbar
plotting.plot_glass_brain(localizer_tmap_filename, threshold=3,
                          colorbar=True, plot_abs=False)


###############################################################################
# Different projections for the left and right hemispheres
# ---------------------------------------------------------
#
# Hemispheric sagittal cuts
plotting.plot_glass_brain(localizer_tmap_filename,
                          title='plot_glass_brain with display_mode="lzr"',
                          black_bg=True, display_mode='lzr', threshold=3)

###############################################################################
plotting.plot_glass_brain(localizer_tmap_filename, threshold=0, colorbar=True,
                          title='plot_glass_brain with display_mode="lyrz"',
                          plot_abs=False, display_mode='lyrz')

plotting.show()
