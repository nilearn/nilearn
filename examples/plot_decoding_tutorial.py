"""
A introduction tutorial to fMRI decoding
==========================================

Here is a simple tutorial on decoding with nilearn. It reproduces the
Haxby 2001 study on a face vs cat discrimination task in a mask of the
ventral stream.

This tutorial is meant as an introduction to the various steps of a
decoding analysis.

It is not a minimalistic example, as it strives to be didactic. It is not
meant to be copied to analyze new data: many of the steps are unecessary.

.. contents:: **Contents**
    :local:
    :depth: 1


"""

###########################################################################
# Retrieve and load the fMRI data from the  Haxby study
# -----------------------------------------------------
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
print('First subject functional nifti images (4D) are at: %s' %
      fmri_filename)  # 4D data

###########################################################################
# Convert the fMRI volume's to a data matrix
# ..........................................
#
# We will use the :class:`nilearn.input_data.NiftiMasker` to extract the
# fMRI data on a mask and convert it to data series.
#
# The mask is a mask of the Ventral Temporal streaming coming from the
# Haxby study:
mask_filename = haxby_dataset.mask_vt[0]

# Let's visualize it, using the subject's anatomical image as a
# background
from nilearn import plotting
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                 cmap='Paired')

###########################################################################
# Now we use the NiftiMasker.
#
# We first create a masker, giving it the options that we care
# about. Here we use standardizing of the data, as it is often important
# for decoding
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)

# We give the masker a filename and retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_filename)

###########################################################################
# The variable "fmri_masked" is a numpy array:
print(fmri_masked)

###########################################################################
# Its shape corresponds to the number of time-points times the number of
# voxels in the mask
print(fmri_masked.shape)

###########################################################################
# Load the behavioral labels
# ..........................
#
# The behavioral labels are stored in a CSV file, separated by spaces.
#
# We use pandas to load them in an array.
import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
print(behavioral)

###########################################################################
# Retrieve the experimental conditions, that we are going to use as
# prediction targets in the decoding
conditions = behavioral['labels']
print(conditions)

###########################################################################
# Restrict the analysis to cats and faces
# ........................................
#
# As we can see from the targets above, the experiment contains many
# conditions, not all that interest us for decoding.
#
# To keep only data corresponding to faces or cats, we create a
# mask of the samples belonging to the condition.
condition_mask = conditions.isin(['face', 'cat'])

# We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
fmri_masked = fmri_masked[condition_mask]

###########################################################################
# We now have less samples
print(fmri_masked.shape)

###########################################################################
# We apply the same mask to the targets
conditions = conditions[condition_mask]
print(conditions.shape)


###########################################################################
# Decoding with an SVM
# ----------------------
#
# We will now use the `scikit-learn <http://www.scikit-learn.org>`_
# machine-learning toolbox on the fmri_masked data.
#
# As a decoder, we use a Support Vector Classification, with a linear
# kernel.
#
# We first create it:
from sklearn.svm import SVC
svc = SVC(kernel='linear')
print(svc)

###########################################################################
# The svc object is an object that can be fit (or trained) on data with
# labels, and then predict labels on data without.
#
# We first fit it on the data
svc.fit(fmri_masked, conditions)

###########################################################################
# We can then predict the labels from the data
prediction = svc.predict(fmri_masked)
print(prediction)

###########################################################################
# Let's measure the error rate:
print((prediction == conditions).sum() / float(len(conditions)))

###########################################################################
# This error rate is meaningless. Why?

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
svc.fit(fmri_masked[:-30], conditions[:-30])

prediction = svc.predict(fmri_masked[-30:])
print((prediction == conditions[-30:]).sum() / float(len(conditions[-30:])))


###########################################################################
# Implementing a KFold loop
# .........................
#
# We can split the data in train and test set repetitively in a `KFold`
# strategy:
from sklearn.cross_validation import KFold

cv = KFold(n=len(fmri_masked), n_folds=5)

for train, test in cv:
    conditions_masked = conditions.values[train]
    svc.fit(fmri_masked[train], conditions_masked)
    prediction = svc.predict(fmri_masked[test])
    print((prediction == conditions.values[test]).sum()
           / float(len(conditions.values[test])))

###########################################################################
# Cross-validation with scikit-learn
# ...................................
#
# Scikit-learn has tools to perform cross-validation easier:
from sklearn.cross_validation import cross_val_score
cv_score = cross_val_score(svc, fmri_masked, conditions)
print(cv_score)

###########################################################################
# Note that we can speed things up to use all the CPUs of our computer
# with the n_jobs parameter.
#
# By default, cross_val_score uses a 3-fold KFold. We can control this by
# passing the "cv" object, here a 5-fold:
cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)

###########################################################################
# The best way to do cross-validation is to respect the structure of
# the experiment, for instance by leaving out full sessions of
# acquisition.
#
# The number of the session is stored in the CSV file giving the
# behavioral data. We have to apply our session mask, to select only cats
# and faces. To leave a session out, we pass it to a
# LeaveOneLabelOut object:
session_label = behavioral['chunks'][condition_mask]

from sklearn.cross_validation import LeaveOneLabelOut
cv = LeaveOneLabelOut(session_label)
cv_score = cross_val_score(svc, fmri_masked, conditions, cv=cv)
print(cv_score)


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
coef_ = svc.coef_
print(coef_)

###########################################################################
# It's a numpy array
print(coef_.shape)

###########################################################################
# We need to turn it back into a Nifti image, in essence, "inverting"
# what the NiftiMasker has done.
#
# For this, we can call inverse_transform on the NiftiMasker:
coef_img = masker.inverse_transform(coef_)
print(coef_img)

###########################################################################
# coef_img is now a NiftiImage.
#
# We can save the coefficients as a nii.gz file:
coef_img.to_filename('haxby_svc_weights.nii.gz')

###########################################################################
# Plotting the SVM weights
# .........................
#
# We can plot the weights, using the subject's anatomical as a background
from nilearn.plotting import plot_stat_map, show

plot_stat_map(coef_img, bg_img=haxby_dataset.anat[0],
              title="SVM weights", display_mode="yx")

show()

###########################################################################
# Further reading
# ----------------
#
# * The :ref:`section of the documentation on decoding <decoding>`
#
# * :ref:`sphx_glr_auto_examples_02_decoding_plot_haxby_anova_svm.py`
#   For decoding without a precomputed mask
#
# * :ref:`space_net`
#
# ______________
