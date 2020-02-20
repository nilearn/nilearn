"""
A introduction tutorial to fMRI decoding
==========================================

Here is a simple tutorial on decoding with nilearn. It reproduces the
Haxby 2001 study on a face vs cat discrimination task in a mask of the
ventral stream.

This tutorial is meant as an introduction to the various steps of a
decoding analysis.

It is not a minimalistic example, as it strives to be didactic. It is not
meant to be copied to analyze new data: many of the steps are unnecessary.

.. contents:: **Contents**
    :local:
    :depth: 1


"""

###########################################################################
# Retrieve and load the fMRI data from the  Haxby study
# ------------------------------------------------------
#
# First download the data
# ........................
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
# Visualizing the fmri volume
# ............................
#
# One way to visualize a fmri volume is using :func:`nilearn.plotting.plot_epi`.
# We will visualize the previously fetched fmri data from Haxby dataset.
#
# Because fmri data is 4D (it consists of many 3D EPI images), we cannot 
# plot it directly using :func:`nilearn.plotting.plot_epi` (which accepts 
# just 3D input). Here we are using :func:`nilearn.image.mean_img` to 
# extract a single 3D EPI image from the fmri data.
#
from nilearn import plotting
from nilearn.image import mean_img
plotting.view_img(mean_img(fmri_filename), threshold=None)

###########################################################################
# Feature extraction: from fMRI volumes to a data matrix
# .......................................................
#
# These are some really lovely images, but for machine learning we need 
# matrices to work with the actual data. To transform our Nifti images into
# matrices, we will use the :class:`nilearn.input_data.NiftiMasker` to 
# extract the fMRI data on a mask and convert it to data series.
#
# A mask of the Ventral Temporal (VT) cortex coming from the
# Haxby study is available:
mask_filename = haxby_dataset.mask_vt[0]

# Let's visualize it, using the subject's anatomical image as a
# background
plotting.plot_roi(mask_filename, bg_img=haxby_dataset.anat[0],
                  cmap='Paired')

###########################################################################
# Now we use the NiftiMasker.
#
# We first create a masker, and ask it to normalize the data to improve the
# decoding. The masker will extract a 2D array ready for machine learning
# with nilearn:
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
fmri_masked = masker.fit_transform(fmri_filename)

###########################################################################
# .. seealso::
# 	You can ask the NiftiMasker to derive a mask given the data. In
# 	this case, it is interesting to have a look at a report to see the
# 	computed mask by using `masker.generate_report`.
masker.generate_report()

###########################################################################
# The variable "fmri_masked" is a numpy array:
print(fmri_masked)

###########################################################################
# Its shape corresponds to the number of time-points times the number of
# voxels in the mask
print(fmri_masked.shape)

###########################################################################
# One way to think about what just happened is to look at it visually:
#
# .. image:: /images/masking.jpg
#
# Essentially, we can think about overlaying a 3D grid on an image. Then,
# our mask tells us which cubes or "voxels" (like 3D pixels) to sample from.
# Since our Nifti images are 4D files, we can't overlay a single grid --
# instead, we use a series of 3D grids (one for each volume in the 4D file),
# so we can get a measurement for each voxel at each timepoint. These are
# reflected in the shape of the matrix ! You can check this by checking the
# number of non-negative voxels in our binary brain mask.
#
# .. seealso::
# 	There are many other strategies in Nilearn :ref:`for masking data and for
# 	generating masks <computing_and_applying_mask>`
# 	I'd encourage you to spend some time exploring the documentation for these.
# 	We can also `display this time series :ref:`sphx_glr_auto_examples_03_connectivity_plot_sphere_based_connectome.py` to get an intuition of how the
# 	whole brain signal is changing over time.
#
# We'll display the first three voxels by sub-selecting values from the
# matrix. You can also find more information on how to slice arrays `here
# <https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#basic-slicing-and-indexing>`_.
import matplotlib.pyplot as plt
plt.plot(fmri_masked[5:150, :3])

plt.title('Voxel Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.tight_layout()

###########################################################################
# Load the behavioral labels
# ...........................
#
# Now that the brain images are converted to a data matrix, we can apply 
# machine-learning to them, for instance to predict the task that the subject 
# was doing. The behavioral labels are stored in a CSV file, separated by
# spaces.
#
# We use pandas to load them in an array.
import pandas as pd
# Load behavioral information
behavioral = pd.read_csv(haxby_dataset.session_target[0], delimiter=' ')
print(behavioral)

###########################################################################
# The task was a visual-recognition task, and the labels denote the 
# experimental condition: the type of object that was presented to the 
# subject. This is what we are going to try to predict.
conditions = behavioral['labels']
conditions

###########################################################################
# Restrict the analysis to cats and faces
# ........................................
#
# As we can see from the targets above, the experiment contains many
# conditions. As a consequence the data is quite big:
print(fmri_masked.shape)

###########################################################################
# Not all of this data has an interest to us for decoding, so we will keep
# only fmri signals corresponding to faces or cats. We create a mask of
# the samples belonging to the condition; this mask is then applied to the
# fmri data to restrict the classification to the face vs cat discrimination.
# As a consequence, the input data is much small (i.e. fmri signal is shorter):
condition_mask = conditions.isin(['face', 'cat'])

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
# ---------------------
#
# As a decoder, we use a Support Vector Classification, with a linear kernel. We
# first create it using by using :class:`nilearn.decoding.Decoder`.
from nilearn.decoding import Decoder
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
import numpy as np
n_samples = len(conditions)
fmri_niimgs_train = index_img(fmri_niimgs, np.arange(0, n_samples - 30))
fmri_niimgs_test = index_img(fmri_niimgs, np.arange(n_samples - 30, n_samples))
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
# We can split the data in train and test set repetitively in a `KFold`
# strategy:
from sklearn.model_selection import KFold
cv = KFold(n_splits=5)

# The "cv" object's split method can now accept data and create a
# generator which can yield the splits.
fold = 0
for train, test in cv.split(conditions):
    fold += 1
    decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True)
    decoder.fit(index_img(fmri_niimgs, train), conditions[train])
    prediction = decoder.predict(index_img(fmri_niimgs, test))
    print("CV Fold {:01d} | Prediction Accuracy: {:.3f}".format(fold, 
        (prediction == conditions[test]).sum() / float(len(conditions[test]))))

###########################################################################
# Cross-validation with the decoder
# ...................................
#
# The decoder implements a cross-validation loop by default, it also returns
# an array of shape (cross-validation parameters, n_folds)..
decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True, cv=cv)
decoder.fit(fmri_niimgs, conditions)

print(decoder.cv_scores_['face'])
print(decoder.cv_scores_['cat'])

# The decoder also gives the best performing parameters per fold.
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
from sklearn.model_selection import LeaveOneGroupOut
cv = LeaveOneGroupOut()

decoder = Decoder(estimator='svc', mask=mask_filename, standardize=True, cv=cv)
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
