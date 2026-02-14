"""
Advanced decoding using scikit learn
====================================

This tutorial opens the box of decoding pipelines to bridge integrated
functionalities provided by the :class:`~nilearn.decoding.Decoder` object
with more advanced usecases. It reproduces basic examples functionalities with
direct calls to scikit-learn function and gives pointers to more advanced
objects. If some concepts seem unclear,
please refer to the :ref:`documentation on decoding <decoding_intro>`
and in particular to the :ref:`advanced section <going_further>`.
As in many other examples, we perform decoding of the visual category of a
stimuli on :footcite:t:`Haxby2001` dataset,
focusing on distinguishing two categories:
face and cat images.

.. include:: ../../../examples/masker_note.rst

"""

# %%
# Retrieve and load the :term:`fMRI` data from the Haxby study
# ------------------------------------------------------------
#
# First download the data
# .......................
#

# The :func:`~nilearn.datasets.fetch_haxby` function will download the
# Haxby dataset composed of fMRI images in a Niimg,
# a spatial mask and a text document with label of each image
from nilearn import datasets

haxby_dataset = datasets.fetch_haxby()
mask_filename = haxby_dataset.mask_vt[0]
fmri_filename = haxby_dataset.func[0]

# Loading the behavioral labels
import pandas as pd

behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
behavioral

# %%
# We keep only a images from a pair of conditions(cats versus faces).
from sklearn.preprocessing import LabelEncoder

conditions = behavioral["labels"]
conditions = conditions.to_numpy()
le = LabelEncoder()
conditions = le.fit_transform(conditions)
run_label = behavioral["chunks"]


# %%
# Masking the data
# ................
# To use a scikit-learn estimator on brain images, you should first mask the
# data using a :class:`~nilearn.maskers.NiftiMasker` to extract only the
# voxels inside the mask of interest,
# and transform 4D input :term:`fMRI` data to 2D arrays
# (`shape=(n_timepoints, n_voxels)`) that estimators can work on.
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(
    mask_img=mask_filename,
    runs=run_label,
    smoothing_fwhm=4,
    standardize="zscore_sample",
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)
fmri_masked = masker.fit_transform(fmri_filename)

# %% convert to torch tensor for GPU processing
import cupy as cp

fmri_masked_cp = cp.asarray(fmri_masked)
conditions_cp = cp.asarray(conditions)

# %%
# Fit the classifier
# ......................
from sklearn import config_context
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_predict

# %%
# Fit the classifier on GPU
# ......................
with config_context(array_api_dispatch=True):
    ridge = RidgeClassifier(solver="svd")
    cv_predict = cross_val_predict(ridge, fmri_masked_cp, conditions_cp, cv=5)

# %%
# Fit the classifier on CPU
# ......................
ridge = RidgeClassifier(solver="svd")
cv_predict = cross_val_predict(ridge, fmri_masked, conditions, cv=5)
