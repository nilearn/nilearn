"""
Copying headers from input images with ``math_img``
===================================================

This example shows how to copy the header information from one of
the input images to the result image when using the function
:func:`~nilearn.image.math_img`.

The header information contains metadata about the image, such as the
dimensions, the voxel sizes, the affine matrix, repetition time (:term:`TR`),
etc. Some of this information might be important for downstream analyses
depending on the software used.
"""

# %%
# Let's fetch an example :term:`fMRI` dataset
from nilearn import datasets

dataset = datasets.fetch_adhd(n_subjects=2)

# %%
# Now let's look at the header of one of these images
from nilearn.image import load_img

subj1_img = load_img(dataset.func[0])
subj2_img = load_img(dataset.func[1])

print(f"Subject 1 header:\n{subj1_img.header}")
# %%
# Let's apply a simple operation using :func:`~nilearn.image.math_img`
from nilearn.image import math_img

result_img = math_img("img1 * 1", img1=subj1_img)

# %%
# By default, :func:`~nilearn.image.math_img` simply resets result image's
# header to the default :class:`~nibabel.nifti1.Nifti1Header`.
#
# This means that it will contain different information as compared to the
# input image.
#
# We can check that as follows:
print("Following header fields do not match:")
for key in result_img.header:
    if not (subj1_img.header[key] == result_img.header[key]).all():
        print(
            f"For '{key}'\n",
            "\tinput image:",
            subj1_img.header[key],
            "\n\tresult image:",
            result_img.header[key],
        )

# %%
# This could affect some downstream analyses.
#
# For example, here the :term:`TR` (given as fifth element in ``pixdim``)
# is changed from 2 in ``subj1_img`` to 1 in ``result_img``.
#
# To fix this, we can copy the header of the input images to the
# result image, like this:
result_img_with_header = math_img(
    "img1 * 1", img1=subj1_img, copy_header_from="img1"
)

# %%
# Let's compare the header fields again.
print("Following header fields do not match:")
for key in result_img_with_header.header:
    if not (subj1_img.header[key] == result_img_with_header.header[key]).all():
        print(
            f"For '{key}'\n",
            "\tinput image:",
            subj1_img.header[key],
            "\n\tresult image:",
            result_img_with_header.header[key],
        )
# %%
# We can safely ignore the fields that are still different -- ``scl_scope`` and
# ``scl_inter`` are just ``nan`` and ``cal_max`` is supposed to have the
# maximum data value that is updated automatically by nilearn.

# %%
# Modifying dimensions in the formula
# -----------------------------------
#
# Now let's say we have a formula that changes the dimensions of the
# input images. For example, by taking the mean of the images along the
# time axis.
#
# Copying the header with the ``copy_header_from`` parameter will not work
# in this case.
#
# So, in such cases we could just use :func:`~nilearn.image.math_img` without
# specifying ``copy_header_from`` and then explicitly copy the header from one
# of the images using :func:`~nilearn.image.new_img_like`
result_img = math_img(
    "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)",
    img1=subj1_img,
    img2=subj2_img,
)

# %%
# Several of the header fields are different:
print("Following header fields do not match:")
for key in result_img.header:
    if not (subj1_img.header[key] == result_img.header[key]).all():
        print(
            f"For '{key}'\n",
            "\tinput image:",
            subj1_img.header[key],
            "\n\tresult image:",
            result_img.header[key],
        )
# %%
# Now we can copy the header explicitly like this:
from nilearn.image import new_img_like

result_img_with_header = new_img_like(
    ref_niimg=subj1_img,
    data=result_img.get_fdata(),
    affine=result_img.affine,
    copy_header=True,
)

# %%
# Now, only a few not-so-important fields are different.
#
# The modified fields can vary depending upon the formula passed into the
# function.
#
# In this case, ``dim`` and ``pixdim`` are different because we took a mean
# over the time dimension.
#
# And again, ``cal_min`` and ``cal_max`` are set to minimum and maximum data
# values respectively, by Nilearn.
print("Following header fields do not match:")
for key in result_img_with_header.header:
    if not (subj1_img.header[key] == result_img_with_header.header[key]).all():
        print(
            f"For '{key}'\n",
            "\tinput image:",
            subj1_img.header[key],
            "\n\tresult image:",
            result_img_with_header.header[key],
        )
