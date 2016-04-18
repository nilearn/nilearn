"""
Negating an image with math_img
===============================

The goal of this example is to illustrate the use of the function
:func:`nilearn.image.math_img` on T-maps.
We compute a negative image by multiplying its voxel values with -1.
"""

from nilearn import datasets, plotting, image

###############################################################################
# Retrieve the data: the localizer dataset with contrast maps.
localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=2,
    get_anats=True,
    get_tmaps=True)
localizer_anat_filename = localizer_dataset.anats[1]
localizer_tmap_filename = localizer_dataset.tmaps[1]

###############################################################################
# Multiply voxel values by -1.
negative_stat_img = image.math_img("-img", img=localizer_tmap_filename)

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="t-map, dim=-.5",
                       dim=-.5)
plotting.plot_stat_map(negative_stat_img,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3, title="Negative t-map, dim=-.5",
                       dim=-.5)
plotting.show()
