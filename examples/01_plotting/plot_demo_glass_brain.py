"""
Glass brain plotting in nilearn
===============================

See :ref:`plotting` for more plotting functionalities.
"""


###############################################################################
# Retrieve data from Internet
# ---------------------------

from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_button_task()
localizer_tmap_filename = localizer_dataset.tmaps[0]

###############################################################################
# Glass brain plotting: whole brain sagittal cuts
# -----------------------------------------------

from nilearn import plotting

plotting.plot_glass_brain(localizer_tmap_filename, threshold=3)

###############################################################################
# Glass brain plotting: black backgrond
# -------------------------------------
# On a black background (option "black_bg"), and with only the x and
# the z view (option "display_mode").
plotting.plot_glass_brain(
    localizer_tmap_filename, title='plot_glass_brain',
    black_bg=True, display_mode='xz', threshold=3)

###############################################################################
# Glass brain plotting: Hemispheric sagittal cuts
# -----------------------------------------------
plotting.plot_glass_brain(localizer_tmap_filename,
                          title='plot_glass_brain with display_mode="lyrz"',
                          display_mode='lyrz', threshold=3)

plotting.show()
