"""
Searchlight analysis of face vs house recognition
==================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

"""

### Load Haxby dataset ########################################################
from nisl import datasets
import numpy as np
import nibabel

dataset_files = datasets.fetch_haxby_simple()

# fmri_data and mask are copied to lose the reference to the original data
bold_img = nibabel.load(dataset_files.func)
fmri_data = np.copy(bold_img.get_data())
affine = bold_img.get_affine()
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
mask = dataset_files.mask

### Preprocess data ###########################################################
# Build the mean image because we have no anatomic data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]

### Loading step ##############################################################
from nisl.io import NiftiMasker
from nibabel import Nifti1Image

nifti_masker = NiftiMasker(mask=mask, sessions=session)
niimg = Nifti1Image(X, affine)
X_masked = nifti_masker.fit(niimg).transform(niimg)
X_preprocessed = nifti_masker.inverse_transform(X_masked).get_data()
X_preprocessed = np.rollaxis(X_preprocessed, axis=-1)
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

### Prepare the masks #########################################################
# Here we will use several masks :
# * mask is the originalmask
# * process_mask is a subset of mask, it contains voxels that should be
#   processed (we only keep the slice z = 26 and the back of the brain to speed
#   up computation)
process_mask = mask.copy()
process_mask[..., 38:] = False
process_mask[..., :36] = False
process_mask[:, 30:] = False

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

searchlight.fit(X_preprocessed, y)

### Visualization #############################################################
import pylab as pl
pl.figure(1)
# searchlight.scores_ contains per voxel cross validation scores
s_scores = np.ma.array(searchlight.scores_, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_img[..., 37]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.imshow(np.rot90(s_scores[..., 37]), interpolation='nearest',
        cmap=pl.cm.hot, vmax=1)
pl.axis('off')
pl.title('Searchlight')
pl.show()

### Show the F_score
from sklearn.feature_selection import f_classif
pl.figure(2)
X_masked = X_preprocessed[:, process_mask]
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
pl.title('F-scores')
pl.axis('off')
pl.show()
