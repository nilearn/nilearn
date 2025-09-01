"""
Searchlight analysis of face vs house recognition
=================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the :term:`fMRI` (see the generated figures).

.. include:: ../../../examples/masker_note.rst

"""

# %%
# Load Haxby dataset
# ------------------
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.image import get_data, load_img, new_img_like

# We fetch 2nd subject from haxby datasets (which is default)
haxby_dataset = fetch_haxby()

# print basic information on the dataset
print(f"Anatomical nifti image (3D) is located at: {haxby_dataset.mask}")
print(f"Functional nifti image (4D) is located at: {haxby_dataset.func[0]}")

fmri_filename = haxby_dataset.func[0]
labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels["labels"]
run = labels["chunks"]

# %%
# Restrict to faces and houses
# ----------------------------
from nilearn.image import index_img

condition_mask = y.isin(["face", "house"])

fmri_img = index_img(fmri_filename, condition_mask)
y, run = y[condition_mask], run[condition_mask]

# %%
# Prepare masks
# -------------
# - mask_img is the original mask
# - process_mask_img is a subset of mask_img, it contains the voxels that
#   should be processed (we only keep the slice z = 29 and the back of the
#   brain to speed up computation)
import numpy as np

mask_img = load_img(haxby_dataset.mask)

# .astype() makes a copy.
process_mask = get_data(mask_img).astype(int)
picked_slice = 29
process_mask[..., (picked_slice + 1) :] = 0
process_mask[..., :picked_slice] = 0
process_mask[:, 30:] = 0
process_mask_img = new_img_like(mask_img, process_mask)

# %%
# Searchlight computation
# -----------------------
# Make processing parallel
#
# .. warning::
#
#     As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 2

# %%
# Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the run, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
#
# The radius is the one of the Searchlight sphere that will scan the volume
from sklearn.model_selection import KFold

import nilearn.decoding

cv = KFold(n_splits=4)

searchlight = nilearn.decoding.SearchLight(
    mask_img,
    process_mask_img=process_mask_img,
    radius=5.6,
    n_jobs=n_jobs,
    verbose=1,
    cv=cv,
)
searchlight.fit(fmri_img, y)

# %%
# Visualization
# -------------
# %%
# After fitting the searchlight, we can access the searchlight scores
# as a NIfTI image using the `scores_img_` attribute.
scores_img = searchlight.scores_img_

# %%
# Use the :term:`fMRI` mean image as a surrogate of anatomical data
from nilearn.image import mean_img

mean_fmri = mean_img(fmri_img, copy_header=True)

# %%
# Because scores are not a zero-center test statistics,
# we cannot use plot_stat_map
from nilearn.plotting import plot_img, plot_stat_map, show

plot_img(
    scores_img,
    bg_img=mean_fmri,
    title="Searchlight scores image",
    display_mode="z",
    cut_coords=[-9],
    vmin=0.2,
    cmap="inferno",
    threshold=0.2,
    black_bg=True,
    colorbar=True,
)

# %%
# F-scores computation
# --------------------
from sklearn.feature_selection import f_classif

from nilearn.maskers import NiftiMasker

# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(
    mask_img=mask_img,
    runs=run,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
)
fmri_masked = nifti_masker.fit_transform(fmri_img)

_, p_values = f_classif(fmri_masked, y)
p_values = -np.log10(p_values)
p_values[p_values > 10] = 10
p_unmasked = get_data(nifti_masker.inverse_transform(p_values))

# F_score results
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
f_score_img = new_img_like(mean_fmri, p_ma)
plot_stat_map(
    f_score_img,
    mean_fmri,
    title="F-scores",
    display_mode="z",
    cut_coords=[-9],
    cmap="inferno",
)

show()

# sphinx_gallery_dummy_images=2
