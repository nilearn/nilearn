"""
Advanced decoding using scikit-learn
====================================

This tutorial opens the box of decoding pipelines, beyond the
functionalities provided by the
:class:`~nilearn.decoding.Decoder` object.
First, we reproduce basic functionalities of the
:class:`~nilearn.decoding.Decoder`
object via direct calls to the underlying scikit-learn functions.
Next, we give pointers towards integrating other scikit-learn
estimators directly.

If some concepts seem unclear,
please refer to the :ref:`documentation on decoding <decoding_intro>`
and in particular to the :ref:`advanced section <going_further>`.
As in many other examples, we decode the visual category of
stimuli in the :footcite:t:`Haxby2001` dataset,
focusing on distinguishing two categories:
"face" and "cat" images.

"""

# %%
# Retrieve and load the :term:`fMRI` data from the Haxby study
# ------------------------------------------------------------
#
# Download the data
# .................
# The :func:`~nilearn.datasets.fetch_haxby` function will download the
# Haxby dataset object, whose attributes include
# the fMRI images as Niimg objects (``func``),
# a spatial mask (``mask_vt``),
# and a CSV with the visual category label for each image (``session_target``).

from nilearn import datasets

haxby_dataset = datasets.fetch_haxby()
mask_filename = haxby_dataset.mask_vt[0]
fmri_filename = haxby_dataset.func[0]

# Loading the behavioral labels
import pandas as pd

behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
behavioral

# %%
# We keep only a images from the conditions of interest ("cat" and "face").
from nilearn.image import index_img

conditions = behavioral["labels"]
condition_mask = conditions.isin(["face", "cat"])
fmri_niimgs = index_img(fmri_filename, condition_mask)
conditions = conditions[condition_mask]
conditions = conditions.to_numpy()
run_label = behavioral["chunks"][condition_mask]

# %%
# Performing decoding with scikit-learn
# -------------------------------------

# %%
# Importing a classifier
# ......................
# We can import many predictive models from scikit-learn that can be used in a
# decoding pipelines.
# They all support a ``.fit()`` method.
# Let's define a Support Vector Classifier
# (or :sklearn:`SVC <modules/svm.html>`).

from sklearn.svm import SVC

svc = SVC()

# %%
# Masking the data
# ................
# To use a scikit-learn estimator on brain images, you should first mask the
# data using a :class:`~nilearn.maskers.NiftiMasker` to extract only the
# voxels inside the mask of interest,
# and transform 4D input :term:`fMRI` data to 2D arrays of
# shape `(n_samples, n_features)` that scikit-learn estimators can work on.
# In our case, this means extracting arrays of
# shape `(n_timepoints, n_voxels)`.
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
fmri_masked = masker.fit_transform(fmri_niimgs)

# %%
# Cross-validation with scikit-learn
# ..................................
# To train and test the model in a meaningful way we use cross-validation with
# the function :func:`sklearn.model_selection.cross_val_score` that computes
# the score for each of the different cross-validation folds.
from sklearn.model_selection import cross_val_score

# Here `cv=5` stipulates a 5-fold cross-validation
cv_scores = cross_val_score(svc, fmri_masked, conditions, cv=5)
print(f"SVC accuracy: {cv_scores.mean():.3f}")

# %%
# Tuning cross-validation parameters
# ..................................
# You can change many parameters of the cross_validation, such as:
#
# * using a different
#   :sklearn:`cross-validation scheme <modules/cross_validation.html>`.
#
# * speeding up the computation by using `n_jobs = -1`, which will spread the
#   computation equally across all processors.
#
# * use a different scoring function, as a keyword or imported from
#   :sklearn:`SVC <modules/model_evaluation.html>`;
#   for example, :func:`sklearn.metrics.roc_auc_score`.
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()
cv_scores = cross_val_score(
    svc,
    fmri_masked,
    conditions,
    cv=cv,
    scoring="roc_auc",
    groups=run_label,
    n_jobs=2,
)
print(f"SVC accuracy (tuned parameters): {cv_scores.mean():.3f}")

# %%
# Measuring the chance level
# --------------------------
# :class:`sklearn.dummy.DummyClassifier` (purely random) estimators are the
# simplest way to measure prediction performance at chance. A more controlled
# way, but slower, is to do permutation testing on the labels, with
# :func:`sklearn.model_selection.permutation_test_score`.

# %%
# Dummy estimator
# ...............
from sklearn.dummy import DummyClassifier

null_cv_scores = cross_val_score(
    DummyClassifier(), fmri_masked, conditions, cv=cv, groups=run_label
)

print(f"Dummy accuracy: {null_cv_scores.mean():.3f}")

# %%
# Permutation test
# ................
from sklearn.model_selection import permutation_test_score

null_cv_scores = permutation_test_score(
    svc, fmri_masked, conditions, cv=cv, groups=run_label
)[1]
print(f"Permutation test score: {null_cv_scores.mean():.3f}")

# %%
# Decoding without a mask: ANOVA-SVM in scikit-learn
# --------------------------------------------------
# We can also implement feature selection before decoding.
# To perform the feature selection, we need to import
# the :mod:`sklearn.feature_selection` module and use
# :func:`sklearn.feature_selection.f_classif`, a simple F-score
# based feature selection (a.k.a.
# `ANOVA <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_).
#
# We can then chain both steps (feature selection and decoding)
# into one composite estimator using
# a :class:`~sklearn.pipeline.Pipeline` object.
# Pipeline objects have several useful properties, as described in
# the :sklearn:`scikit-learn documentation <modules/compose.html>`.
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

feature_selection = SelectPercentile(f_classif, percentile=10)
linear_svc = LinearSVC(dual=True, random_state=0)
anova_svc = Pipeline([("anova", feature_selection), ("svc", linear_svc)])

# %%
# We can now use our Pipeline ``anova_svc`` object exactly
# as we were using our ``svc`` estimator previously.
# Previously, we used :func:`sklearn.model_selection.cross_val_score`
# to return the cross-validated decoding scores.
# However, we now want to investigate our model's feature selection
# via its weights.
# We can use :func:`sklearn.model_selection.cross_validate` function
# with ``return_estimator = True`` to save the estimator.
from sklearn.model_selection import cross_validate

fitted_pipeline = cross_validate(
    anova_svc,
    fmri_masked,
    conditions,
    cv=cv,
    groups=run_label,
    return_estimator=True,
)
print(f"ANOVA+SVC test score: {fitted_pipeline['test_score'].mean():.3f}")

# %%
# Visualize the :term:`ANOVA` + SVC's discriminating weights
# ..........................................................
# First, we retrieve the Pipeline object fitted on the first
# cross-validation fold and its SVC coefficients.

first_pipeline = fitted_pipeline["estimator"][0]
svc_coef = first_pipeline.named_steps["svc"].coef_
print(
    "After feature selection, "
    f"the SVC is trained only on {svc_coef.shape[1]} features"
)

# %%
# Next, we use the ``inverse_transform`` function to
# invert the feature selection step
# and put these coefficients in the right place in
# our `(n_timepoints, n_voxels)` 2D array.
full_coef = first_pipeline.named_steps["anova"].inverse_transform(svc_coef)

print(
    "After inverting feature selection, "
    f"we have {full_coef.shape[1]} features back"
)

# %%
# Finally, we apply the ``inverse_transform`` function
# of our :class:`~nilearn.maskers.NiftiMasker` object
# to re-create a 4D Niimg that we can visualize.
from nilearn.plotting import plot_stat_map, show

weight_img = masker.inverse_transform(full_coef)
plot_stat_map(weight_img, title="ANOVA+SVC weights", draw_cross=False)

show()

# %%
# Going further with scikit-learn
# -------------------------------
# While the above analysis mirrored what occurs in
# the :class:`~nilearn.decoding.Decoder` object,
# we can go still further with scikit-learn.
# Two examples are given below, but many more are possible.

# %%
# Changing the prediction engine
# ..............................
# To change the prediction engine, we just need to import it and use in our
# pipeline instead of the SVC.
# For example, we can try Fisher's
# :sklearn:`Linear Discriminant Analysis (LDA)
# <auto_examples/decomposition/plot_pca_vs_lda.html>`.

# Construct the new estimator object and use it in a new Pipeline
# after feature-selection with ANOVA, as before
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

feature_selection = SelectPercentile(f_classif, percentile=10)
lda = LinearDiscriminantAnalysis()
anova_lda = Pipeline([("anova", feature_selection), ("LDA", lda)])

# Recompute the cross-validation score:
import numpy as np

cv_scores = cross_val_score(
    anova_lda, fmri_masked, conditions, cv=cv, groups=run_label
)
classification_accuracy = np.mean(cv_scores)
n_conditions = len(set(conditions))  # number of target classes
print(
    f"ANOVA + LDA classification accuracy: {classification_accuracy:.4f} "
    f"/ Chance Level: {1.0 / n_conditions:.4f}"
)

# %%
# Changing the feature selection
# ..............................
# Let's say that you want a more sophisticated feature selection;
# for example,
# a Recursive Feature Elimination (:class:`~sklearn.feature_selection.RFE`)
# before a SVC.
# We can simply follow the same principle as we did in changing
# the prediction engine.

from sklearn.feature_selection import RFE

svc = SVC()
rfe = RFE(SVC(kernel="linear", C=1.0), n_features_to_select=50, step=0.25)

# Create a new pipeline, composing the two classifiers `rfe` and `svc`.

rfe_svc = Pipeline([("rfe", rfe), ("svc", svc)])

# Recompute the cross-validation score
# cv_scores = cross_val_score(rfe_svc,
#                             fmri_masked,
#                             target,
#                             cv=cv,
#                             n_jobs=2,
#                             verbose=1)
# But, be aware that this can take some time....

# %%
# References
# ----------
#
# .. footbibliography::
