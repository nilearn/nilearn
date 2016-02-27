"""
============================================================
Encoding models for visual stimuli from Miyawaki et al. 2008
============================================================

This example demonstrates how to build an encoding model for
functional MRI data with visual stimuli.

Encoding models try to predict neuronal activity using properties of
a real world stimulus, like images or sound.

In the dataset from :func:`nilearn.datasets.fetch_miyawaki2008`,
participants were shown images consisting of 10x10
binary (either black or white) pixels and the corresponding  fMRI activity
was recorded. We will predict the neuronal activity from the pixel-values
of the presented images for each voxel. Then we extract the receptive fields
for a set of voxels to see which pixel location a voxel is most sensitive to.

See also :doc:`plot_miyawaki_reconstruction` for a decoding
approach for the same dataset.
"""

##############################################################################
# Loading the data
# ----------------
# First some imports:

import numpy as np
import pylab as plt
from nilearn.plotting import plot_stat_map
from nilearn import masking


##############################################################################
# Now we can load the data set:
from nilearn.datasets import fetch_miyawaki2008

dataset = fetch_miyawaki2008()

##############################################################################
# We only use the training data of this study,
# where random binary images were shown.

X_random_filenames = dataset.func[12:]
y_random_filenames = dataset.label[12:]

# shape of the binary image in pixels
y_shape = (10, 10)


##############################################################################
# We can use :func:`nilearn.input_data.MultiNiftiMaser` to load the fMRI data,
# clean and mask it.

from nilearn.input_data import MultiNiftiMasker

masker = MultiNiftiMasker(mask_img=dataset.mask, detrend=True,
                          standardize=True)
masker.fit()
X_train = masker.transform(X_random_filenames)

# We load the visual stimuli from csv files
y_train = []
for y in y_random_filenames:
    y_train.append(np.reshape(np.loadtxt(y, dtype=np.int, delimiter=','),
                              (-1,) + y_shape, order='F'))


##############################################################################
# We now stack the X and y data and remove an offset in the beginning/end.

X_train = np.vstack([x[2:] for x in X_train])
y_train = np.vstack([y[:-2] for y in y_train]).astype(float)


##############################################################################
# X_train is a matrix of *N_samples* x *N_voxels*

print(X_train.shape)

##############################################################################
# We flatten the last two dimensions of y_train
# so it is a matrix of *N_samples* x *N_pixels*.

# Flatten the stimuli
y_train = np.reshape(y_train, (-1, y_shape[0] * y_shape[1]))

print(y_train.shape)

##############################################################################
# Building the encoding models
# ----------------------------
# We can now proceed to do a simple encoding using Ridge regression.

from sklearn.linear_model import Ridge
from sklearn.cross_validation import KFold

##############################################################################
# Using 10-fold cross-validation, we partition the data into 10 'Folds',
# we hold out each fold of the data for testing, then fit a ridge regression
# to the remaining 9/10 of the data, using y_train as predictors
# and X_train as targets, and create predictions for the held-out 10th.

estimator = Ridge(alpha=100.)
cv = KFold(len(y_train), 10)

predictions = [Ridge(alpha=100.).fit(
    y_train.reshape(-1, 100)[train], X_train[train]).predict(
        y_train.reshape(-1, 100)[test])
    for train, test in cv]


##############################################################################
# To have a measure for the quality of our encoding model, we estimate how
# much variance our encoding model explains in each voxel.

scores = [1. - (((X_train[test] - pred) ** 2).sum(axis=0) /
         ((X_train[test] - X_train[test].mean(axis=0)) ** 2).sum(axis=0))
         for pred, (train, test) in zip(predictions, cv)]

##############################################################################
# Mapping the encoding scores on the brain
# ----------------------------------------
# To plot the scores onto the brain, we use :func:`nilearn.image.new_img_like`
# to create a Nifti1Image containing the scores and then threshold it:

from nilearn.image import threshold_img, new_img_like
from nilearn.image.resampling import coord_transform

cut_score = np.array(scores).mean(axis=0)
cut_score[cut_score < 0] = 0

# bring the scores into the shape of the background brain
score_map = new_img_like(dataset.background,
                         masking.unmask(cut_score, dataset.mask).get_data())

thresholded_score_map = threshold_img(score_map, threshold = 1e-6)

##############################################################################
# Plotting the statistical map on a background brain, we mark four voxels
# which we will inspect more closely later on.

def index_to_xy_coord(x,y,z=10):
    '''Transforms data index to coordinates of the background + offset'''
    coords = coord_transform(x,y,z,affine=thresholded_score_map.get_affine())
    return coords + np.array([0,1,0])


xy_indices_of_special_voxels = [(30, 10), (32, 10), (31, 9), (31, 10)]

statmap = plot_stat_map(thresholded_score_map, bg_img=dataset.background,
                        cut_coords=[-8], display_mode='z', aspect=1.25,
                        title='Explained variance per voxel',
                        symmetric_cbar=False)

# creating a contour for each voxel and adding it to the statistical map

for i, (x, y) in enumerate(xy_indices_of_special_voxels):
    statmap.add_markers(np.array(index_to_xy_coord(x,y))[np.newaxis,:],
                        edgecolor=['r', 'b', 'magenta','g'][i],
                        marker_size=150, marker = 's',
                        facecolor = 'none', lw = 4.5)


# re-set figure size after construction so colorbar gets rescaled too
fig = plt.gcf()
fig.set_size_inches(12,12)

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
# pixels, so we use Lasso regression to estimate a sparse set of
# regression coefficients.

from sklearn.linear_model import LassoLarsCV

# automatically estimate the sparsity by cross-validation
lasso = LassoLarsCV(max_iter=10)

# Mark the same pixel in each receptive field
p = (4,2)

import matplotlib as mpl
from matplotlib import gridspec

fig = plt.figure(figsize=(12,12))
gs1 = gridspec.GridSpec(2, 3)

# we fit the Lasso for each of the three voxels of the upper row
for i, index in enumerate([1780, 1951, 2131]):
    ax = plt.subplot(gs1[0,i])
    # we reshape the coefficients into the form of the original images
    rf = lasso.fit(y_train, X_train[:, index]).coef_.reshape((10, 10))
    # add a black background
    ax.imshow(np.zeros_like(rf), vmin=0., vmax=1., cmap='gray')
    ax_im = ax.imshow(np.ma.masked_less(rf, 0.1), interpolation="nearest",
                      cmap=['Reds','Greens','Blues'][i],vmin=0.,vmax=0.75)
    # add the marked pixel
    ax.add_patch(mpl.patches.Rectangle(
        (p[1] - .5, p[0] - .5), 1, 1,
        facecolor = 'none', edgecolor = 'r', lw = 4))
    plt.axis('off')
    plt.colorbar(ax_im, ax=ax)

# and then for the voxel at the bottom

gs1.update(left=0., right=1., wspace=0.1, bottom = 0.3)
ax = plt.subplot(gs1[1,1])
# we reshape the coefficients into the form of the original images
rf = lasso.fit(y_train, X_train[:, 1935]).coef_.reshape((10, 10))
ax.imshow(np.zeros_like(rf), vmin=0., vmax=1., cmap='gray')
ax_im = ax.imshow(np.ma.masked_less(rf, 0.1), interpolation="nearest",
                  cmap='RdPu',vmin=0.,vmax=0.75)

# add the marked pixel
ax.add_patch(mpl.patches.Rectangle(
    (p[1] - .5, p[0] - .5), 1, 1,
    facecolor = 'none', edgecolor = 'r', lw = 4))
plt.axis('off')
plt.colorbar(ax_im, ax=ax)
##############################################################################
# The receptive fields of the four voxels are not only close to each other,
# the relative location of the pixel each voxel is most sensitive to
# roughly maps to the relative location of the voxels to each other.
