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
# Load the behavioral labels
# ..........................
#
# The behavioral labels are stored in a CSV file, separated by spaces.
#
# We use numpy to load them in an array.
import numpy as np
# Load behavioral information
behavioral = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
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
condition_mask = np.logical_or(conditions == b'face', conditions == b'cat')

# We apply this mask in the sampe direction to restrict the
# classification to the face vs cat discrimination
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily.
from nilearn.image import index_img
fmri_niimgs = index_img(fmri_filename, condition_mask)

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
# We will use The mask is a mask of the Ventral Temporal streaming coming from
# the Haxby study.
#
# We first create it:
from nilearn.decoding import Decoder
mask_filename = haxby_dataset.mask_vt[0]
decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True)

###########################################################################
# The svc object is an object that can be fit (or trained) on data with
# labels, and then predict labels on data without.
#
# We first fit it on the data
decoder.fit(fmri_niimgs, conditions)

###########################################################################
# We can then predict the labels from the data
prediction = decoder.predict(fmri_niimgs)
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
n_samples = len(conditions)
fmri_niimgs_train = index_img(fmri_niimgs, np.arange(0, n_samples - 30))
fmri_niimgs_test = index_img(fmri_niimgs, np.arange(n_samples - 30, n_samples))
conditions_train = conditions[:-30]
conditions_test = conditions[-30:]

decoder.fit(fmri_niimgs_train, conditions_train)

prediction = decoder.predict(fmri_niimgs_test)
print((prediction == conditions_test).sum() / float(len(conditions_test)))


###########################################################################
# Cross-validation with the decoder
# ...................................
#
# The best way to do cross-validation is to respect the structure of
# the experiment, for instance by leaving out full sessions of
 # acquisition.
# The decoder implements a cross-validation loop by default, it also returns
# an array of shape (cross-validation parameters, n_folds)..
print(decoder.cv_scores_)

# The decoder also gives the best performing parameters per fold.
print(decoder.cv_params_)


# from sklearn.cross_validation import KFold

# cv = KFold(n=len(fmri_masked), n_folds=5)

# from sklearn.cross_validation import LeaveOneLabelOut
# cv = LeaveOneLabelOut(session_label)

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
# It's a numpy array
print(coef_.shape)
# To get the nifti image of these coeficients, we only need retrieve the
# coef_img_ in the decoder and select the class
print(decoder.coef_img_)
# We can save the coefficients as a nii.gz file:
decoder.coef_img_['face'].to_filename('haxby_svc_weights.nii.gz')

###########################################################################
# Plotting the SVM weights
# .........................
#
# We can plot the weights, using the subject's anatomical as a background
from nilearn.plotting import plot_stat_map, show
plot_stat_map(decoder.coef_img_['face'], bg_img=haxby_dataset.anat[0],
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

