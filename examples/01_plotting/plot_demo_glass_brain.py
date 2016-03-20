"""
Glass brain plotting in nilearn
===============================

See :ref:`plotting` for more plotting functionalities.
"""


###############################################################################
# Retrieve the data
from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=2,
    get_tmaps=True)
localizer_tmap_filename = localizer_dataset.tmaps[1]

###############################################################################
# Demo glass brain plotting. Whole brain sagittal cuts
from nilearn import plotting

plotting.plot_glass_brain(localizer_tmap_filename, threshold=3)

###############################################################################
# On a black background (option "black_bg"), and with only the x and
# the z view (option "display_mode").
plotting.plot_glass_brain(
    localizer_tmap_filename, title='plot_glass_brain',
    black_bg=True, display_mode='xz', threshold=3)

plotting.show()

###############################################################################
# Hemispheric sagittal cuts
plotting.plot_glass_brain(
    localizer_tmap_filename, display_mode='lyrz', threshold=3)

plotting.show()
