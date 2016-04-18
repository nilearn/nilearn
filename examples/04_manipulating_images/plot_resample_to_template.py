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
# Now, the localizer t-map image can be resampled to the MNI template image.
from nilearn.image import resample_to_img

resampled_localizer_tmap = resample_to_img(localizer_tmap_filename, template)

###############################################################################
# Let's check the shape and affine have been correctly updated.
from nilearn.image import index_img

original_shape = index_img([localizer_dataset.tmaps[0]], 0).shape
original_affine = index_img([localizer_dataset.tmaps[0]], 0).get_affine()

resampled_shape = resampled_localizer_tmap.shape
resampled_affine = resampled_localizer_tmap.get_affine()

template_shape = index_img([template], 0).shape
template_affine = index_img([template], 0).get_affine()
print("""Shape comparison:
- Original t-map image shape : {0}
- Resampled t-map image shape: {1}
- Template image shape       : {2}
""".format(original_shape, resampled_shape, template_shape))

print("""Affine comparison:
- Original t-map image affine :\n {0}
- Resampled t-map image affine:\n {1}
- Template image affine       :\n {2}
""".format(original_affine, resampled_affine, template_affine))

###############################################################################
# Finally, result images are displayed using nilearn plotting module.
from nilearn import plotting

plotting.plot_stat_map(localizer_tmap_filename,
                       bg_img=localizer_anat_filename,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="t-map on original anat")
plotting.plot_stat_map(resampled_localizer_tmap,
                       bg_img=template,
                       cut_coords=(36, -27, 66),
                       threshold=3,
                       title="Resampled t-map on MNI template anat")
plotting.show()
