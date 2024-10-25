"""Skull extraction of whole-brain MNI ICBM152 1mm-resolution T1 template and \
rescale+typecasting+compression of the whole-brain, grey-matter and \
white-matter MNI ICBM152 1mm-resolution templates.

This script takes as inputs the original MNI ICBM152 1mm-resolution templates
 and the corresponding 'whole-brain' mask. They can be fetched from the OSF
 (Open Science Framework) Nilearn account: https://osf.io/7pj92/download

This script outputs the templates that are loaded by the following Nilearn
 functions:
 nilearn.datasets.load_mni152_template
 nilearn.datasets.load_mni152_gm_template
 nilearn.datasets.load_mni152_wm_template

Compatibility: Nilearn 0.7.1, Python 3.7.3
"""

# Authors: Ana Luisa Pinho, Nicolas Gensollen, Jerome Dockes

import gzip
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nilearn.image import get_data, load_img, new_img_like
from nilearn.masking import apply_mask, unmask
from nilearn.plotting import plot_img

# Inputs
templates_paths = [
    "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz",
    "mni_icbm152_gm_tal_nlin_sym_09a.nii.gz",
    "mni_icbm152_wm_tal_nlin_sym_09a.nii.gz",
]

brain_mask = load_img("mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz")


for template_path in templates_paths:
    # Load template
    template = load_img(template_path)
    plot_img(template, colorbar=True)

    # Remove skull of whole-brain template
    if template_path == "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz":
        niimg = unmask(apply_mask(template, brain_mask), brain_mask)
        # plot_img(niimg, colorbar=True)
    else:
        niimg = template

    # Re-scale
    new_data = get_data(niimg)
    new_data /= np.max(new_data)
    new_data *= 255
    new = new_img_like(template, new_data)
    # plot_img(new, colorbar=True)

    # Change re-scaled image from numpy.float64 to numpy.uint8
    new_img = new_img_like(new, get_data(new).astype("uint8"))
    # plot_img(new_img, colorbar=True)

    # Store and gzip with maximum compression rate
    fname_nii = Path(template_path.split(".", 1)[0] + "_converted.nii")
    new_img.to_filename(fname_nii)
    fname_nii_gz = fname_nii.with_suffix(f"{fname_nii.suffix}.gz")
    with fname_nii.open("rb") as f_in:
        content = f_in.read()
        with gzip.GzipFile(
            filename=fname_nii_gz, mode="w", compresslevel=9
        ) as f_out:
            f_out.write(content)

    # Check compressed file
    nifti_image = load_img(fname_nii_gz)
    plot_img(nifti_image, colorbar=True)

    plt.show()
