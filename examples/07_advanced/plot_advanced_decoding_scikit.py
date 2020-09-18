"""
Advanced decoding using scikit learn
==========================================

This tutorial opens the box of decoding pipelines to bridge integrated
functionalities provided by the :class:`nilearn.decoding.Decoder` object
with more advanced usecases. It reproduces basic examples functionalities with
direct calls to scikit-learn function and gives pointers to more advanced
objects. If some concepts seem unclear, please refer to the :ref:`documentation on
decoding <decoding_intro>` and in particular to the :ref:`advanced section <going_further>`.
As in many other examples, we perform decoding of the visual category of a stimuli on Haxby
2001 dataset, focusing on distinguishing two categories : face and cat images.

    * J.V. Haxby et al. "Distributed and Overlapping Representations of Faces
      and Objects in Ventral Temporal Cortex", Science vol 293 (2001), p
      2425.-2430.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

###########################################################################
# Retrieve and load the fMRI data from the Haxby study
# ------------------------------------------------------
#
# First download the data
# ........................
#
#
# doctest: +SKIP
from sklearn.feature_selection import RFE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nilearn.plotting import plot_stat_map
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import permutation_test_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import KFold
from nilearn.decoding import Decoder
from nilearn.image import index_img
from nilearn.image import mean_img
from nilearn import plotting
from nilearn import datasets
import pandas as pd

# The :func:`nilearn.datasets.fetch_haxby` function will download the
# Haxby dataset composed of fmri images in a Niimg, a spatial mask and a text
# document with label of each image
haxby_dataset = datasets.fetch_haxby()
mask_filename = haxby_dataset.mask_vt[0]
fmri_filename = haxby_dataset.func[0]
# Loading the behavioral labels
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')

# We keep only a images from a pair of conditions(cats versus faces).
conditions = behavioral['labels']
condition_mask = conditions.isin(['face', 'cat'])
fmri_niimgs = index_img(fmri_filename, condition_mask)
conditions = conditions[condition_mask]
# Convert to numpy array
conditions = conditions.values
session_label = behavioral['chunks'][condition_mask]

###########################################################################
# Performing decoding with scikit-learn
# --------------------------------------
# Importing a classifier
# ........................
# We can import many predictive models from scikit-learn that can be used in a
# decoding pipelines. They are all used with the same `fit()` and `predict()`
# functions. Let's define a `Support Vector Classifier<http://scikit-learn.org/stable/modules/svm.html >`_(or SVC).

svc = SVC()

###########################################################################
# Masking the data
# ...................................
# To use a scikit- learn estimator on brain images, you should first mask the
# data using a: class: `nilearn.input_data.NiftiMasker`: to extract only the
# voxels inside the mask of interest, and transform 4D input fMRI data to
# 2D arrays(shape(n_timepoints, n_voxels)) that estimators can work on.

masker = NiftiMasker(mask_img=mask_filename, sessions=session_label,
                     smoothing_fwhm=4, standardize=True,
                     memory="nilearn_cache", memory_level=1)
fmri_masked = masker.fit_transform(fmri_niimgs)

###########################################################################
# Cross-validation with scikit-learn
# ...................................
# To train and test the model in a meaningful way we use cross-validation with
# the function: func: `sklearn.model_selection.cross_val_score` that computes
# for you the score for the different folds of cross-validation.

cv_scores = cross_val_score(svc, fmri_masked, conditions, cv=5)
# Here `cv=5` stipulates a 5-fold cross-validation

###########################################################################
# Tuning cross-validation parameters
# ...................................
# You can change many parameters of the cross_validation here, for example:
#
# * use a different cross - validation scheme, for example LeaveOneGroupOut()
#
# * speed up the computation by using n_jobs = -1, which will spread the
#   computation equally across all processors.
#
# * use a different scoring function, as a keyword or imported from scikit-learn
#   scoring = 'roc_auc'

cv = LeaveOneGroupOut()
cv_scores = cross_val_score(svc, fmri_masked, conditions, cv=cv, scoring='roc_auc',
                            groups=session_label, n_jobs=-1)

###########################################################################
# Measuring the chance level
# ------------------------------------
# :class:`sklearn.dummy.DummyClassifier` (purely random) estimators are the
# simplest way to measure prediction performance at chance. A more controlled
# way, but slower, is to do permutation testing on the labels, with
# :func:`sklearn.model_selection.permutation_test_score`.

###########################################################################
# Dummy estimator
# ...................................

null_cv_scores = cross_val_score(
    DummyClassifier(), fmri_masked, conditions, cv=cv)
print("Dummy accuracy: {:.3f}".format(null_cv_scores))

###########################################################################
# Permutation test
# ...................................

null_cv_scores = permutation_test_score(svc, fmri_masked, conditions, cv=cv)
print("Permutation test score: {:.3f}".format(null_cv_scores))


###########################################################################
# Decoding without a mask: Anova-SVM in scikit-lean
# --------------------------------------------------
# We can also implement feature selection before decoding as a scikit-learn
# `pipeline`(:class:`sklearn.pipeline.Pipeline`). For this, we need to import
# the :mod:`sklearn.feature_selection` module and use
# :func:`sklearn.feature_selection.f_classif`, a simple F-score
# based feature selection (a.k.a. `Anova <https://en.wikipedia.org/wiki/Analysis_of_variance#The_F-test>`_),

feature_selection = SelectPercentile(f_classif, percentile=5)
anova_svc = Pipeline([('anova', feature_selection), ('svc', svc)])
# We can use our ``anova_svc`` object exactly as we were using our ``svc``
# object previously.
cv_scores = cross_val_score(anova_svc, fmri_masked, conditions,
                            cv=cv, groups=session_label)
print(cv_scores.mean())

###########################################################################
# Visualize the SVC's discriminating weights
# ...........................................

coef = svc.coef_
# We apply back the feature selection to put the coefs in the right 2D place
coef = feature_selection.inverse_transform(coef)
# We apply the inverse of masking to make a 4D image that we can plot
weight_img = masker.inverse_transform(coef)
plot_stat_map(weight_img, title='Anova+SVC weights')


# The decoder object is an object that can be fit (or trained) on data with
# labels, and then predict labels on data without.
#
# We first fit it on the data
decoder.fit(fmri_niimgs, conditions)


###########################################################################
# Going further with scikit-learn
# ------------------------------------

###########################################################################
# Changing the prediction engine
# ...............................
# To change the prediction engine, we just need to import it and use in our
# pipeline instead of the SVC. We can try Fisher's `Linear Discriminant Analysis (LDA) <http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html>`_


# Construct the new estimator object and use it in a new pipeline after anova
feature_selection = SelectPercentile(f_classif, percentile=5)
lda = LinearDiscriminantAnalysis()
anova_lda = Pipeline([('anova', feature_selection), ('LDA', lda)])

# Recompute the cross-validation score::

cv_scores = cross_val_score(anova_lda, fmri_masked, target, cv=cv, verbose=1)
classification_accuracy = np.mean(cv_scores)
n_conditions = len(set(target))  # number of target classes
print("Classification accuracy: %.4f / Chance Level: %.4f" %
      (classification_accuracy, 1. / n_conditions))

###########################################################################
# Changing the feature selection
# ...........................................
# Let's say that you want a more sophisticated feature selection, for example a
# Recursive Feature Elimination(RFE) before a svc. We follows the same principle.

# Import it and define your fancy objects
svc = SVC()
rfe = RFE(SVC(kernel='linear', C=1.), 50, step=0.25)

# Create a new pipeline, composing the two classifiers `rfe` and `svc`: :

rfe_svc = Pipeline([('rfe', rfe), ('svc', svc)])

# Recompute the cross-validation score:
# cv_scores = cross_val_score(rfe_svc, fmri_masked, target, cv=cv, n_jobs=-1, verbose=1)
# But, be aware that this can take * A WHILE * ...

###########################################################################
# Note that for this classification task both classes contain the same number
# of samples (the problem is balanced). Then, we can use accuracy to measure
# the performance of the decoder. This is done by defining accuracy as the
# `scoring`.
# Let's measure the prediction accuracy:
print((prediction == conditions).sum() / float(len(conditions)))

###########################################################################
# This prediction accuracy score is meaningless. Why?

###########################################################################
# Measuring prediction scores using cross-validation
# ---------------------------------------------------
#
# The proper way to measure error rates or prediction accuracy is via
# cross-validation: leaving out some data and testing on it.
#
# Manually leaving out data
# ..........................
#
# Let's leave out the 30 last data points during training, and test the
# prediction on these 30 last points:
fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -30))
fmri_niimgs_test = index_img(fmri_niimgs, slice(-30, None))
conditions_train = conditions[:-30]
conditions_test = conditions[-30:]

decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True)
decoder.fit(fmri_niimgs_train, conditions_train)

prediction = decoder.predict(fmri_niimgs_test)

# The prediction accuracy is calculated on the test data: this is the accuracy
# of our model on examples it hasn't seen to examine how well the model perform
# in general.

print("Prediction Accuracy: {:.3f}".format(
    (prediction == conditions_test).sum() / float(len(conditions_test))))

###########################################################################
# Implementing a KFold loop
# ..........................
#
# We can manually split the data in train and test set repetitively in a
# `KFold` strategy by importing scikit-learn's object:
cv = KFold(n_splits=5)

# The "cv" object's split method can now accept data and create a
# generator which can yield the splits.
fold = 0
for train, test in cv.split(conditions):
    fold += 1
    decoder = Decoder(estimator='svc', mask=mask_filename,
                      standardize=True)
    decoder.fit(index_img(fmri_niimgs, train), conditions[train])
    prediction = decoder.predict(index_img(fmri_niimgs, test))
    print(
        "CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(
            fold,
            (prediction == conditions[test]).sum() / float(len(
                conditions[test]))))

###########################################################################
# Cross-validation with the decoder
# ...................................
#
# The decoder also implements a cross-validation loop by default and returns
# an array of shape (cross-validation parameters, `n_folds`). We can use
# accuracy score to measure its performance by defining `accuracy` as the
# `scoring` parameter.
n_folds = 5
decoder = Decoder(
    estimator='svc', mask=mask_filename,
    standardize=True, cv=n_folds,
    scoring='accuracy'
)
decoder.fit(fmri_niimgs, conditions)

###########################################################################
# Cross-validation pipeline can also be implemented manually. More details can
# be found on `scikit-learn website
# <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html`_.
#
# Then we can check the best performing parameters per fold.
print(decoder.cv_params_['face'])

###########################################################################
# .. note::
# 	We can speed things up to use all the CPUs of our computer with the
# 	n_jobs parameter.
#
# The best way to do cross-validation is to respect the structure of
# the experiment, for instance by leaving out full sessions of
# acquisition.
#
# The number of the session is stored in the CSV file giving the
# behavioral data. We have to apply our session mask, to select only cats
# and faces.
session_label = behavioral['chunks'][condition_mask]

###########################################################################
# The fMRI data is acquired by sessions, and the noise is autocorrelated in a
# given session. Hence, it is better to predict across sessions when doing
# cross-validation. To leave a session out, pass the cross-validator object
# to the cv parameter of decoder.
cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True,
                  cv=cv)
decoder.fit(fmri_niimgs, conditions, groups=session_label)

print(decoder.cv_scores_)

###########################################################################
# Inspecting the model weights
# -----------------------------
#
# Finally, it may be useful to inspect and display the model weights.
#
# Turning the weights into a nifti image
# .......................................
#
# We retrieve the SVC discriminating weights
coef_ = decoder.coef_
print(coef_)

###########################################################################
# It's a numpy array with only one coefficient per voxel:
print(coef_.shape)

###########################################################################
# To get the Nifti image of these coefficients, we only need retrieve the
# `coef_img_` in the decoder and select the class

coef_img = decoder.coef_img_['face']

###########################################################################
# coef_img is now a NiftiImage.  We can save the coefficients as a nii.gz file:
decoder.coef_img_['face'].to_filename('haxby_svc_weights.nii.gz')

###########################################################################
# Plotting the SVM weights
# .........................
#
# We can plot the weights, using the subject's anatomical as a background
plotting.view_img(
    decoder.coef_img_['face'], bg_img=haxby_dataset.anat[0],
    title="SVM weights", dim=-1
)

###########################################################################
# Further reading
# ----------------
#
# * The :ref:`section of the documentation on decoding <decoding`
#
# * :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py`
#   For decoding without a precomputed mask
#
# * :ref:`frem`
#
# * :ref:`space_net`
#
# ______________
