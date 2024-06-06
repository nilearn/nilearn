"""Understanding :class:`nilearn.decoding.Decoder`
==================================================

Nilearn's :class:`nilearn.decoding.Decoder` object is a composite estimator
that does several things under the hood and can hence be a bit difficult to
understand at first.

This example aims to provide a clear understanding of the
:class:`nilearn.decoding.Decoder` object by demonstrating these steps via a
Sklearn pipeline.

We will use the :footcite:t:`Haxby2001` dataset where the participants were
shown images of 8 different types as described in the
:ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py` example.
We will train a classifier to predict the label of the object in the stimulus
image based on the subject's fMRI data from the Ventral Temporal cortex.
"""

# %%
# Load the Haxby dataset
# ----------------------
from nilearn import datasets

# By default 2nd subject data will be fetched on which we run our analysis
haxby_dataset = datasets.fetch_haxby()
fmri_img = haxby_dataset.func[0]
# Pick the mask that we will use to extract the data from Ventral Temporal
# cortex
mask_vt = haxby_dataset.mask_vt[0]

import numpy as np

# Load the behavioral data
import pandas as pd

from nilearn.image import index_img

behavioral_data = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
labels = behavioral_data["labels"]
# Keep the trials corresponding to all the labels except the ``rest`` ones
labels_mask = labels != "rest"
y = labels[labels_mask]
y = y.values

# Load run information
run = behavioral_data["chunks"][labels_mask]
run = run.values

# Also keep the fmri data corresponding to these labels
fmri_img = index_img(fmri_img, labels_mask)

# Overview of the input data
print(f"{len(np.unique(y))} labels to predict (y): {np.unique(y)}")
print(f"fMRI data shape (X): {fmri_img.shape}")
print(f"Runs (groups): {np.unique(run)}")
# %%
# Preprocess the fMRI data
# ------------------------
#
# As we can see, the fMRI data is a 4D image with shape (40, 64, 64, 864).
# Here 40x64x64 are the dimensions of the 3D brain image and 864 is the number
# of trials, each corresponding to one of the 8 labels we selected above.
#
# :class:`nilearn.decoding.Decoder` can convert this 4D image to a 2D numpy
# array where each row corresponds to a trial and each column corresponds to a
# voxel. In addition, it can also do several other things like masking,
# smoothing, standardizing the data etc. depending on your requirements.
#
# Under the hood, it uses :class:`nilearn.maskers.NiftiMasker` to do all these
# operations. So here we will demonstrate this by directly using the
# :class:`nilearn.maskers.NiftiMasker`. We will use it to:
#   1. only keep the data from the Ventral Temporal cortex by providing the
#      mask image (in :class:`nilearn.decoding.Decoder` this is done by
#      providing the mask image in the ``mask`` parameter).
#   2. standardize the data by z-scoring it such that the data is scaled to
#      have zero mean and unit variance across trials (in
#      :class:`nilearn.decoding.Decoder` this is done by setting the
#      ``standardize`` parameter to ``"zscore_sample"``).

from nilearn.maskers import NiftiMasker

masker = NiftiMasker(mask_img=mask_vt, standardize="zscore_sample")

# Fit and transform the data with the masker
X = masker.fit_transform(fmri_img)

print(f"fMRI data shape after masking: {X.shape}")
# So now we have a 2D numpy array of shape (864, 464) where each row
# corresponds to a trial and each column corresponds to a feature
# (voxel in the Ventral Temporal cortex).

# %%
# Convert the multi-class labels to binary labels
# -----------------------------------------------
#
# The :class:`nilearn.decoding.Decoder` converts multi-class classification
# problem to N one-vs-others binary classification problems by default, where N
# is the number of unique labels. We have 8 unique labels in our case.

from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
y_binary = label_binarizer.fit_transform(y)

# %%
# Let's plot the labels to understand the conversion
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder

# create a copy of y_binary and manipulate it just for plotting
y_binary_ = y_binary.copy()
for col in range(y_binary_.shape[1]):
    y_binary_[np.where(y_binary_[:, col] == 1), col] = col

fig, (ax_binary, ax_multi) = plt.subplots(
    2, gridspec_kw={"height_ratios": [10, 1.5]}, figsize=(12, 2)
)
cmap = ListedColormap(["white"] + list(plt.cm.tab10.colors)[0:8])
binary_plt = ax_binary.imshow(
    y_binary_.T,
    aspect="auto",
    cmap=cmap,
    interpolation="nearest",
    origin="lower",
)
ax_binary.set_xticks([])
ax_binary.set_yticks([])
ax_binary.set_ylabel("One-vs-Others")
# encode the original labels for plotting
label_multi = LabelEncoder()
y_multi = label_multi.fit_transform(y)
y_multi = y_multi.reshape(1, -1)
cmap = ListedColormap(list(plt.cm.tab10.colors)[0:8])
multi_plt = ax_multi.imshow(
    y_multi,
    aspect="auto",
    interpolation="nearest",
    cmap=cmap,
)
ax_multi.set_yticks([])
ax_multi.set_xlabel("Original trial sequence")
cbar = fig.colorbar(multi_plt, ax=[ax_binary, ax_multi])
cbar.set_ticks(np.arange(1 + len(label_multi.classes_)))
cbar.set_ticklabels([*label_multi.classes_, "all others"])

plt.show()

# %%
# So at the bottom we have the original sequence in which the trials were
# presented and at the top we have the labels in the one-vs-others format.
# The white color corresponds to the trials that are not of the corresponding
# class in the one-vs-others format.
#
# The :class:`nilearn.decoding.Decoder` does this conversion internally and
# considers each of these binary classification problems as a separate problem
# to solve.

# %%
# Hyperparameter optimization
# ---------------------------
#
# The :class:`nilearn.decoding.Decoder` also performs hyperparameter tuning.
# How this is done depends on the estimator used:
#
# Except for the Support Vector Machine classifiers/regressors
# (used by setting ``estimator="svc"`` or ``"svc_l1"`` or ``"svc_l2"`` or
# ``"svr"``), the hyperparameter tuning is done using the ``...CV`` classes
# from Sklearn. This essentially means that the hyperparameters are optimized
# using an internal cross-validation on the training data.
#
# For the SVM classifiers/regressors, the best performing hyperparameters for
# the given train-test splits are picked.
#
# In addition, the parameters grids that are used for hyperparameter tuning
# by :class:`nilearn.decoding.Decoder` are also different from the default
# Sklearn parameters grids for the corresponding ``...CV`` classes.
#
# We can replicate this behavior for later use by defining a function that
# selects the estimator depending on the estimator string provided.

# %%
# ReNA clustering
# ---------------
#
# TODO

# %%
# Feature selection
# -----------------
#
# TODO

# %%
# Decode via an Sklearn pipeline
# ------------------------------
#
# TODO

# %%
# Decode via the :class:`nilearn.decoding.Decoder`
# ------------------------------------------------
#
# TODO

# %%
# Compare the results
# -------------------
#
# TODO
