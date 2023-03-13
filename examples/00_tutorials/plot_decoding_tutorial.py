"""
A introduction tutorial to fMRI decoding
========================================

Here is a simple tutorial on decoding with nilearn. It reproduces the
Haxby 2001 study on a face vs cat discrimination task in a mask of the
ventral stream.

    * J.V. Haxby et al. "Distributed and Overlapping Representations of Faces
      and Objects in Ventral Temporal Cortex", Science vol 293 (2001), p
      2425.-2430.

This tutorial is meant as an introduction to the various steps of a decoding
analysis using Nilearn meta-estimator: :class:`nilearn.decoding.Decoder`

It is not a minimalistic example, as it strives to be didactic. It is not
meant to be copied to analyze new data: many of the steps are unnecessary.
"""


###########################################################################
# Retrieve and load the fMRI data from the Haxby study
# ----------------------------------------------------
#
# First download the data
# .......................
#
# The :func:`nilearn.datasets.fetch_haxby` function will download the
# Haxby dataset if not present on the disk, in the nilearn data directory.
# It can take a while to download about 310 Mo of data from the Internet.
from nilearn import datasets

# By default 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()
# 'func' is a list of filenames: one for each subject
fmri_filename = haxby_dataset.func[0]

# print basic information on the dataset
print(f"First subject functional nifti images (4D) are at: {fmri_filename}")

###########################################################################
# Visualizing the fmri volume
# ...........................
#
# One way to visualize a :term:`fmri<fMRI>` volume is
# using :func:`nilearn.plotting.plot_epi`.
# We will visualize the previously fetched :term:`fmri<fMRI>`
# data from Haxby dataset.
#
# Because :term:`fmri<fMRI>` data are 4D (they consist of many 3D EPI images),
# we cannot plot them directly using :func:`nilearn.plotting.plot_epi`
# (which accepts just 3D input).
# Here we are using :func:`nilearn.image.mean_img` to
# extract a single 3D EPI image from the :term:`fmri<fMRI>` data.
#
from nilearn import plotting
from nilearn.image import mean_img

plotting.view_img(mean_img(fmri_filename), threshold=None)

###########################################################################
# Feature extraction: from fMRI volumes to a data matrix
# ......................................................
#
# These are some really lovely images, but for machine learning
# we need matrices to work with the actual data. Fortunately, the
# :class:`nilearn.decoding.Decoder` object we will use later on can
# automatically transform Nifti images into matrices.
# All we have to do for now is define a mask filename.
#
# A mask of the Ventral Temporal (VT) cortex coming from the
# Haxby study is available:
mask_filename = haxby_dataset.mask_vt[0]

# Let's visualize it, using the subject's anatomical image as a
# background
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0], cmap="Paired")

###########################################################################
# Load the behavioral labels
# ..........................
#
# Now that the brain images are converted to a data matrix, we can apply
# machine-learning to them, for instance to predict the task that the subject
# was doing. The behavioral labels are stored in a CSV file, separated by
# spaces.
#
# We use pandas to load them in an array.
import pandas as pd

# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=" ")
print(behavioral)

###########################################################################
# The task was a visual-recognition task, and the labels denote the
# experimental condition: the type of object that was presented to the
# subject. This is what we are going to try to predict.
conditions = behavioral["labels"]
print(conditions)

###########################################################################
# Restrict the analysis to cats and faces
# .......................................
#
# As we can see from the targets above, the experiment contains many
# conditions. As a consequence, the data is quite big. Not all of this data
# has an interest to us for decoding,
# so we will keep only :term:`fmri<fMRI>` signals
# corresponding to faces or cats.
# We create a mask of the samples belonging to
# the condition; this mask is then applied
# to the :term:`fmri<fMRI>` data to restrict the
# classification to the face vs cat discrimination.
#
# The input data will become much smaller
# (i.e. :term:`fmri<fMRI>` signal is shorter):
condition_mask = conditions.isin(["face", "cat"])

###########################################################################
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily.
from nilearn.image import index_img

fmri_niimgs = index_img(fmri_filename, condition_mask)

###########################################################################
# We apply the same mask to the targets
conditions = conditions[condition_mask]
# Convert to numpy array
conditions = conditions.values
print(conditions.shape)

###########################################################################
# Decoding with Support Vector Machine
# ------------------------------------
#
# As a decoder, we use a Support Vector Classifier with a linear kernel. We
# first create it using by using :class:`nilearn.decoding.Decoder`.
from nilearn.decoding import Decoder

decoder = Decoder(estimator="svc", mask=mask_filename, standardize=True)

###########################################################################
# The decoder object is an object that can be fit (or trained) on data with
# labels, and then predict labels on data without.
#
# We first fit it on the data
decoder.fit(fmri_niimgs, conditions)

###########################################################################
# We can then predict the labels from the data
prediction = decoder.predict(fmri_niimgs)
print(prediction)

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
# --------------------------------------------------
#
# The proper way to measure error rates or prediction accuracy is via
# cross-validation: leaving out some data and testing on it.
#
# Manually leaving out data
# .........................
#
# Let's leave out the 30 last data points during training, and test the
# prediction on these 30 last points:
fmri_niimgs_train = index_img(fmri_niimgs, slice(0, -30))
fmri_niimgs_test = index_img(fmri_niimgs, slice(-30, None))
conditions_train = conditions[:-30]
conditions_test = conditions[-30:]

decoder = Decoder(estimator="svc", mask=mask_filename, standardize=True)
decoder.fit(fmri_niimgs_train, conditions_train)

prediction = decoder.predict(fmri_niimgs_test)

# The prediction accuracy is calculated on the test data: this is the accuracy
# of our model on examples it hasn't seen to examine how well the model perform
# in general.

print(
    "Prediction Accuracy: {:.3f}".format(
        (prediction == conditions_test).sum() / float(len(conditions_test))
    )
)

###########################################################################
# Implementing a KFold loop
# .........................
#
# We can manually split the data in train and test set repetitively in a
# `KFold` strategy by importing scikit-learn's object:
from sklearn.model_selection import KFold

cv = KFold(n_splits=5)

for fold, (train, test) in enumerate(cv.split(conditions), start=1):
    decoder = Decoder(estimator="svc", mask=mask_filename, standardize=True)
    decoder.fit(index_img(fmri_niimgs, train), conditions[train])
    prediction = decoder.predict(index_img(fmri_niimgs, test))
    print(
        "CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(
            fold,
            (prediction == conditions[test]).sum()
            / float(len(conditions[test])),
        )
    )

###########################################################################
# Cross-validation with the decoder
# .................................
#
# The decoder also implements a cross-validation loop by default and returns
# an array of shape (cross-validation parameters, `n_folds`). We can use
# accuracy score to measure its performance by defining `accuracy` as the
# `scoring` parameter.
n_folds = 5
decoder = Decoder(
    estimator="svc",
    mask=mask_filename,
    standardize=True,
    cv=n_folds,
    scoring="accuracy",
)
decoder.fit(fmri_niimgs, conditions)

###########################################################################
# Cross-validation pipeline can also be implemented manually. More details can
# be found on `scikit-learn website
# <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_.
#
# Then we can check the best performing parameters per fold.
print(decoder.cv_params_["face"])

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
session_label = behavioral["chunks"][condition_mask]

###########################################################################
# The :term:`fMRI` data is acquired by sessions,
# and the noise is autocorrelated in a
# given session. Hence, it is better to predict across sessions when doing
# cross-validation. To leave a session out, pass the cross-validator object
# to the cv parameter of decoder.
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()

decoder = Decoder(estimator="svc", mask=mask_filename, standardize=True, cv=cv)
decoder.fit(fmri_niimgs, conditions, groups=session_label)

print(decoder.cv_scores_)

###########################################################################
# Inspecting the model weights
# ----------------------------
#
# Finally, it may be useful to inspect and display the model weights.
#
# Turning the weights into a nifti image
# ......................................
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

coef_img = decoder.coef_img_["face"]

###########################################################################
# coef_img is now a NiftiImage.  We can save the coefficients as a nii.gz file:
decoder.coef_img_["face"].to_filename("haxby_svc_weights.nii.gz")

###########################################################################
# Plotting the SVM weights
# ........................
#
# We can plot the weights, using the subject's anatomical as a background
plotting.view_img(
    decoder.coef_img_["face"],
    bg_img=haxby_dataset.anat[0],
    title="SVM weights",
    dim=-1,
)

###########################################################################
# What is the chance level accuracy?
# ----------------------------------
#
# Does the model above perform better than chance?
# To answer this question, we measure a score at random using simple strategies
# that are implemented in the :class:`nilearn.decoding.Decoder` object. This is
# useful to inspect the decoding performance by comparing to a score at chance.

###########################################################################
# Let's define a object with Dummy estimator replacing 'svc' for classification
# setting. This object initializes estimator with default dummy strategy.
dummy_decoder = Decoder(
    estimator="dummy_classifier", mask=mask_filename, cv=cv
)
dummy_decoder.fit(fmri_niimgs, conditions, groups=session_label)

# Now, we can compare these scores by simply taking a mean over folds
print(dummy_decoder.cv_scores_)

###########################################################################
# Further reading
# ---------------
#
# * The :ref:`section of the documentation on decoding <decoding>`
#
# * :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py`
#   For decoding without a precomputed mask
#
# * :ref:`frem`
#
# * :ref:`space_net`
#
# ______________
