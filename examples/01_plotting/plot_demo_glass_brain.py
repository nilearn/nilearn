"""
Glass brain plotting in nilearn
===============================

See :ref:`plotting` for more plotting functionalities.
"""


###############################################################################
# Load data
# ---------------------------

from nilearn import datasets

stat_img = datasets.load_sample_motor_activation_image()

###############################################################################
# Glass brain plotting: whole brain sagittal cuts
# -----------------------------------------------

from nilearn import plotting

plotting.plot_glass_brain(stat_img, threshold=3)

###############################################################################
# Glass brain plotting: black background
# --------------------------------------
# On a black background (option "black_bg"), and with only the x and
# the z view (option "display_mode").
plotting.plot_glass_brain(
    stat_img, title='plot_glass_brain',
    black_bg=True, display_mode='xz', threshold=3)

###############################################################################
# Glass brain plotting: Hemispheric sagittal cuts
# -----------------------------------------------
plotting.plot_glass_brain(stat_img,
                          title='plot_glass_brain with display_mode="lyrz"',
                          display_mode='lyrz', threshold=3)

plotting.show()
