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
from nilearn.datasets import (
    load_mni152_template,
    load_sample_motor_activation_image,
)

template = load_mni152_template(resolution=2)

stat_img = load_sample_motor_activation_image()

###############################################################################
# Now, the motor contrast map image can be resampled to the MNI template image.
from nilearn.image import resample_to_img

resampled_stat_img = resample_to_img(stat_img, template)

###############################################################################
# Let's check the shape and affine have been correctly updated.

# First load the original t-map in memory:
from nilearn.image import load_img

tmap_img = load_img(stat_img)

original_shape = tmap_img.shape
original_affine = tmap_img.affine

resampled_shape = resampled_stat_img.shape
resampled_affine = resampled_stat_img.affine

template_img = load_img(template)
template_shape = template_img.shape
template_affine = template_img.affine
print(
    f"""Shape comparison:
- Original t-map image shape : {original_shape}
- Resampled t-map image shape: {resampled_shape}
- Template image shape       : {template_shape}
"""
)

print(
    f"""Affine comparison:
- Original t-map image affine :
 {original_affine}
- Resampled t-map image affine:
 {resampled_affine}
- Template image affine       :
 {template_affine}
"""
)

###############################################################################
# Finally, result images are displayed using nilearn plotting module.
from nilearn import plotting

plotting.plot_stat_map(
    stat_img,
    bg_img=template,
    cut_coords=(36, -27, 66),
    threshold=3,
    title="t-map in original resolution",
)
plotting.plot_stat_map(
    resampled_stat_img,
    bg_img=template,
    cut_coords=(36, -27, 66),
    threshold=3,
    title="Resampled t-map",
)
plotting.show()
