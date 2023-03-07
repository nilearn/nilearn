"""
Visualizing global patterns with a carpet plot
==============================================

A common quality control step for functional MRI data is to visualize the data
over time in a carpet plot (also known as a Power plot or a grayplot).

The :func:`nilearn.plotting.plot_carpet()` function generates a carpet plot
from a 4D functional image.
"""

###############################################################################
# Fetching data from ADHD dataset
# -------------------------------
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# plot_carpet can infer TR from the image header, but preprocessing can often
# overwrite that particular header field, so we will be explicit.
t_r = 2.0

# Print basic information on the dataset
print(
    f"First subject functional nifti image (4D) is at: {adhd_dataset.func[0]}"
)

###############################################################################
# Deriving a mask
# ---------------
from nilearn import masking

# Build an EPI-based mask because we have no anatomical data
mask_img = masking.compute_epi_mask(adhd_dataset.func[0])

###############################################################################
# Visualizing global patterns over time
# -------------------------------------
import matplotlib.pyplot as plt
from nilearn.plotting import plot_carpet

display = plot_carpet(adhd_dataset.func[0], mask_img, t_r=t_r)

display.show()

###############################################################################
# Deriving a label-based mask
# ---------------------------
# Create a gray matter/white matter/cerebrospinal fluid mask from
# ICBM152 tissue probability maps.
import numpy as np
from nilearn import image

atlas = datasets.fetch_icbm152_2009()
atlas_img = image.concat_imgs((atlas["gm"], atlas["wm"], atlas["csf"]))
map_labels = {"Gray Matter": 1, "White Matter": 2, "Cerebrospinal Fluid": 3}

atlas_data = atlas_img.get_fdata()
discrete_version = np.argmax(atlas_data, axis=3) + 1
discrete_version[np.max(atlas_data, axis=3) == 0] = 0
discrete_atlas_img = image.new_img_like(atlas_img, discrete_version)


###############################################################################
# Visualizing global patterns, separated by tissue type
# -----------------------------------------------------
from nilearn.plotting import plot_carpet

fig, ax = plt.subplots(figsize=(10, 10))

display = plot_carpet(
    adhd_dataset.func[0],
    discrete_atlas_img,
    t_r=t_r,
    mask_labels=map_labels,
    axes=ax,
    cmap="gray",
)

fig.show()
