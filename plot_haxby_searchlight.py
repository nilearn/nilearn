"""
The haxby dataset: face vs house in object recognition using Searchlight
========================================================================

A significant part of the running time of this example is actually spent
in loading the data: we load all the data but only use the face and
houses conditions.

Plus, in order to speed up computing, Searchlight is run only on one slice
od the fMRI (see the generated figures).
"""

### Load Haxby dataset ########################################################
from nisl import datasets
dataset = datasets.fetch_haxby()
fmri_data = dataset.data
mask = dataset.mask
affine = dataset.affine
y = dataset.target
conditions = dataset.target_strings
session = dataset.session

### Preprocess data ###########################################################
import numpy as np

# Change axis in order to have X under n_samples * x * y * z
X = np.rollaxis(fmri_data, 3)
# X.shape is (1452, 40, 64, 64)

# Mean image: used as background in visualisation
mean_img = np.mean(X, axis=0)

# Detrend data on each session independently
from scipy import signal
for s in np.unique(session):
    X[session == s] = signal.detrend(X[session == s], axis=0)

### Prepare the masks #########################################################
# Here we will use several masks :
# * mask is the originalmask
# * process_mask is a subset of mask, it contains voxels that should be
#   processed (we only keep the slice z = 26 and the back of the brain to speed
#   up computation)
mask = (dataset.mask != 0)
process_mask = mask.copy()
process_mask[..., 38:] = False
process_mask[..., :36] = False
process_mask[:, 30:] = False

### Restrict to faces and houses ##############################################

# Keep only data corresponding to face or houses
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = X[condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

### Searchlight ###############################################################

# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 1

### Define the score function used to evaluate classifiers
# Here we use precision which maesures proportion of true positives among
# all positives results for one class.
from sklearn.metrics import precision_score
score_func = precision_score

### Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.cross_validation import KFold
cv = KFold(y.size, k=4)

### Fit #######################################################################

from nisl import searchlight

# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = searchlight.SearchLight(mask, process_mask, radius=1.5,
        n_jobs=n_jobs, score_func=score_func, verbose=1, cv=cv)

# scores.scores_ is an array containing per voxel cross validation scores
scores = searchlight.fit(X, y)

### Visualization #############################################################
import pylab as pl
pl.figure(1)
s_scores = np.ma.array(scores.scores_, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_img[..., 37]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.imshow(np.rot90(s_scores[..., 37]), interpolation='nearest',
        cmap=pl.cm.hot, vmax=1)
pl.axis('off')
pl.show()

### Show the F_score
from sklearn.feature_selection import f_classif
pl.figure(2)
X_masked = X[:, process_mask]
f_values, p_values = f_classif(X_masked, y)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
p_unmasked = np.zeros(mask.shape)
p_unmasked[process_mask] = p_values
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_img[..., 37]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.imshow(np.rot90(p_ma[..., 37]), interpolation='nearest',
        cmap=pl.cm.hot)
pl.axis('off')
pl.show()
