"""
Resample an image to a template
===============================

The goal of this example is to illustrate the use of the function
:func:`nilearn.image.resample_to_img` to resample an image to a template.
We use the MNI152 template as the reference for resampling a t-map image.
Function :func:`nilearn.image.resample_img` could also be used to achieve this.
"""

###############################################################################
# First we load the required datasets using the nilearn datasets module.
from nilearn.datasets import fetch_localizer_contrasts
from nilearn.datasets import load_mni152_template

template = load_mni152_template()

localizer_dataset = fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=1,
    get_anats=True,
    get_tmaps=True)

localizer_tmap_filename = localizer_dataset.tmaps[0]
localizer_anat_filename = localizer_dataset.anats[0]

###############################################################################
# Now, the localizer t-map can be resampled to the MNI template.
from nilearn.image import resample_to_img
resampled_localizer_tmap = resample_to_img(localizer_tmap_filename, template)

###############################################################################
# Finally, results are displayed using nilearn plotting module.
from nilearn import plotting
plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="t-map on original anat, dim=-.5",
                       dim=-.5)
plotting.plot_stat_map(resampled_localizer_tmap,
                       bg_img=template,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="t-map on MNI template anat, dim=-.5",
                       dim=-.5)
plotting.show()
