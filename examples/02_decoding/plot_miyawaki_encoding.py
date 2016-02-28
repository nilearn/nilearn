"""
============================================================
Encoding models for visual stimuli from Miyawaki et al. 2008
============================================================

This example partly reproduces the encoding model presented in
    `Visual image reconstruction from human brain activity
    using a combination of multiscale local image decoders
    <http://www.cell.com/neuron/abstract/S0896-6273%2808%2900958-6>`_,
    Miyawaki, Y., Uchida, H., Yamashita, O., Sato, M. A.,
    Morito, Y., Tanabe, H. C., ... & Kamitani, Y. (2008).
    Neuron, 60(5), 915-929.

Encoding models try to predict neuronal activity using a presented stimulus,
like an image or sound. Where decoding goes from brain data to
real-world stimulus, encoding goes the other direction.

We demonstrate how to build such an **encoding model** in nilearn, predicting
**fMRI data** from **visual stimuli**, using the dataset from `Miyawaki et al.,
2008  <http://www.cell.com/neuron/abstract/S0896-6273%2808%2900958-6>`_.

In a part of this experiment, participants were shown images consisting of
10x10 binary (either black or white) pixels and the corresponding fMRI
activity was recorded. We will try to predict the activity in each voxel
from the binary pixel-values of the presented images. Then we extract the
receptive fields for a set of voxels to see which pixel location a voxel
is most sensitive to.

See also :doc:`plot_miyawaki_reconstruction` for a decoding
approach for the same dataset.
"""

##############################################################################
# Loading the data
# ----------------
# Some general imports:

import numpy as np
import pylab as plt


##############################################################################
# Now we can load the data set:
from nilearn.datasets import fetch_miyawaki2008

dataset = fetch_miyawaki2008()

##############################################################################
# We only use the training data of this study,
# where random binary images were shown.

# training data starts after the first 12 files
X_random_filenames = dataset.func[12:]
y_random_filenames = dataset.label[12:]

##############################################################################
# We can use :func:`nilearn.input_data.MultiNiftiMasker` to load the fMRI
# data, clean and mask it.

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=dataset.mask, detrend=True,
                          standardize=True)
masker.fit()
X_train = masker.transform(X_random_filenames)

# shape of the binary image in pixels
y_shape = (10, 10)

# We load the visual stimuli from csv files
y_train = []
for y in y_random_filenames:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))

##############################################################################
# Let's take a look at some of these binary images:

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(y_train[0][124], interpolation='nearest', cmap='gray')
plt.axis('off')
plt.title('Run {}, Stimulus {}'.format(1, 125))
plt.subplot(1, 2, 2)
plt.imshow(y_train[2][101], interpolation='nearest', cmap='gray')
plt.axis('off')
plt.title('Run {}, Stimulus {}'.format(3, 102))
plt.subplots_adjust(wspace=0.5)

##############################################################################
# We now stack the X and y data and remove an offset in the beginning/end.

X_train = np.vstack([x[2:] for x in X_train])
y_train = np.vstack([y[:-2] for y in y_train]).astype(float)

##############################################################################
# X_train is a matrix of *samples* x *voxels*

print(X_train.shape)

##############################################################################
# We flatten the last two dimensions of y_train
# so it is a matrix of *samples* x *pixels*.

# Flatten the stimuli
y_train = np.reshape(y_train, (-1, y_shape[0] * y_shape[1]))

print(y_train.shape)

##############################################################################
# Building the encoding models
# ----------------------------
# We can now proceed to build a simple **voxel-wise encoding model** using
# `Ridge regression <http://en.wikipedia.org/wiki/Tikhonov_regularization>`_.
# For each voxel we fit an independent regression model,
# using the pixel-values of the visual stimuli to predict the neuronal
# activity in this voxel.

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

##############################################################################
# Using 10-fold cross-validation, we partition the data into 10 'folds',
# we hold out each fold of the data for testing, then fit a ridge regression
# to the remaining 9/10 of the data, using y_train as predictors
# and X_train as targets, and create predictions for the held-out 10th.

estimator = Ridge(alpha=100.)
cv = KFold(len(y_train), 10)

predictions = []
for train, test in cv:
    predictions.append(Ridge(alpha=100.).fit(
    y_train.reshape(-1, 100)[train], X_train[train]).predict(
        y_train.reshape(-1, 100)[test]))

##############################################################################
# To have a measure for the quality of our encoding model, we estimate how
# much variance the encoding model explains in each voxel.

from sklearn.metrics import r2_score

scores = [r2_score(X_train[test], pred, multioutput = 'raw_values')
         for pred, (train, test) in zip(predictions, cv)]

##############################################################################
# Mapping the encoding scores on the brain
# ----------------------------------------
# To plot the scores onto the brain, we use :func:`nilearn.image.new_img_like`
# to create a Nifti1Image containing the scores and then threshold it:

from nilearn.image import threshold_img, new_img_like
from nilearn.masking import unmask

cut_score = np.array(scores).mean(axis=0)
cut_score[cut_score < 0] = 0

# bring the scores into the shape of the background brain
score_map = new_img_like(dataset.background,
                         unmask(cut_score, dataset.mask).get_data())

thresholded_score_map = threshold_img(score_map, threshold = 1e-6)

##############################################################################
# Plotting the statistical map on a background brain, we mark four voxels
# which we will inspect more closely later on.

from nilearn.plotting import plot_stat_map
from nilearn.image.resampling import coord_transform

def index_to_xy_coord(x, y, z=10):
    '''Transforms data index to coordinates of the background + offset'''
    coords = coord_transform(x, y, z,
                             affine=thresholded_score_map.get_affine())
    return np.array(coords)[np.newaxis, :] + np.array([0, 1, 0])


xy_indices_of_special_voxels = [(30, 10), (32, 10), (31, 9), (31, 10)]

display = plot_stat_map(thresholded_score_map, bg_img=dataset.background,
                        cut_coords=[-8], display_mode='z', aspect=1.25,
                        title='Explained variance per voxel')

# creating a marker for each voxel and adding it to the statistical map

for i, (x, y) in enumerate(xy_indices_of_special_voxels):
    display.add_markers(index_to_xy_coord(x, y),
                        edgecolor=['r', 'b', 'magenta', 'g'][i],
                        marker_size=150, marker = 's',
                        facecolor = 'none', lw = 4.5)


# re-set figure size after construction so colorbar gets rescaled too
fig = plt.gcf()
fig.set_size_inches(12, 12)

plt.show()

##############################################################################
# Estimating receptive fields
# ---------------------------
# Now we take a closer look at the receptive fields of the four marked voxels.
# A voxel's `receptive field <http://en.wikipedia.org/wiki/Receptive_field>`_
# is the region of a stimulus (like an image) where the presence of an object,
# like a white instead of a black pixel, results in a change in activity
# in the voxel. In our case the receptive field is just the vector of 100
# regression  coefficients (one for each pixel) reshaped into the 10x10
# form of the original images. Some voxels are receptive to only very few
# pixels, so we use `Lasso regression
# <http://en.wikipedia.org/wiki/Lasso_(statistics)>`_ to estimate a sparse
# set of regression coefficients.

from sklearn.linear_model import LassoLarsCV

# automatically estimate the sparsity by cross-validation
lasso = LassoLarsCV(max_iter=10)

# Mark the same pixel in each receptive field
marked_pixel = (4, 2)

from matplotlib import gridspec
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(12, 12))
gs1 = gridspec.GridSpec(2, 3)

# we fit the Lasso for each of the three voxels of the upper row
for i, index in enumerate([1780, 1951, 2131]):
    ax = plt.subplot(gs1[0, i])
    # we reshape the coefficients into the form of the original images
    rf = lasso.fit(y_train, X_train[:, index]).coef_.reshape((10, 10))
    # add a black background
    ax.imshow(np.zeros_like(rf), vmin=0., vmax=1., cmap='gray')
    ax_im = ax.imshow(np.ma.masked_less(rf, 0.1), interpolation="nearest",
                      cmap=['Reds', 'Greens', 'Blues'][i], vmin=0., vmax=0.75)
    # add the marked pixel
    ax.add_patch(Rectangle(
        (marked_pixel[1] - .5, marked_pixel[0] - .5), 1, 1,
        facecolor = 'none', edgecolor = 'r', lw = 4))
    plt.axis('off')
    plt.colorbar(ax_im, ax=ax)

# and then for the voxel at the bottom

gs1.update(left=0., right=1., wspace=0.1, bottom = 0.3)
ax = plt.subplot(gs1[1, 1])
# we reshape the coefficients into the form of the original images
rf = lasso.fit(y_train, X_train[:, 1935]).coef_.reshape((10, 10))
ax.imshow(np.zeros_like(rf), vmin=0., vmax=1., cmap='gray')
ax_im = ax.imshow(np.ma.masked_less(rf, 0.1), interpolation="nearest",
                  cmap='RdPu', vmin=0., vmax=0.75)

# add the marked pixel
ax.add_patch(Rectangle(
    (marked_pixel[1] - .5, marked_pixel[0] - .5), 1, 1,
    facecolor = 'none', edgecolor = 'r', lw = 4))
plt.axis('off')
plt.colorbar(ax_im, ax=ax)

##############################################################################
# The receptive fields of the four voxels are not only close to each other,
# the relative location of the pixel each voxel is most sensitive to
# roughly maps to the relative location of the voxels to each other.
