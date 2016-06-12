"""
Controling the contrast of the background when plotting
=========================================================

The `dim` argument controls the contrast of the background.

*dim* modifies the contrast of this image: dim=0 leaves the image
unchanged, negative values of *dim* enhance it, and positive values
decrease it (dim the background).

This *dim* argument may also be useful for the plot_roi function used to
display ROIs on top of a background image.
"""

# Retrieve the data: the localizer dataset with contrast maps

from nilearn import datasets

localizer_dataset = datasets.fetch_localizer_button_task(get_anats=True)
localizer_anat_filename = localizer_dataset.anats[0]
localizer_tmap_filename = localizer_dataset.tmaps[0]

# Plotting: vary the 'dim' of the background
from nilearn import plotting

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="dim=-.5",
                       dim=-.5)

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="dim=0",
                       dim=0)

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="dim=.5",
                       dim=.5)

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="dim=1",
                       dim=1)

plotting.show()
