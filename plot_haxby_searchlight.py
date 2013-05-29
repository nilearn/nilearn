"""
Searchlight analysis of face vs house recognition
==================================================

Searchlight analysis requires fitting a classifier a large amount of
times. As a result, it is an intrinsically slow method. In order to speed
up computing, in this example, Searchlight is run only on one slice on
the fMRI (see the generated figures).

"""

### Load Haxby dataset ########################################################
import numpy as np
import nibabel
from nisl import datasets

dataset_files = datasets.fetch_haxby_simple()

# fmri_data is copied to break the reference to the original data
bold_img = nibabel.load(dataset_files.func)
fmri_data = np.asarray(bold_img.get_data()).copy()
affine = bold_img.get_affine().copy()
del bold_img

y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']

### Preprocess data ###########################################################
# Build the mean image because we have no anatomical data
mean_img = fmri_data.mean(axis=-1)

### Restrict to faces and houses ##############################################
from nibabel import Nifti1Image
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
X = fmri_data[..., condition_mask]
y = y[condition_mask]
session = session[condition_mask]
conditions = conditions[condition_mask]
niimg = Nifti1Image(X, affine)

### Loading step ##############################################################
from nisl.io import NiftiMasker

nifti_masker = NiftiMasker(mask=dataset_files.mask, sessions=session,
                           memory='nisl_cache', memory_level=1)
X_masked = nifti_masker.fit_transform(niimg)

### Prepare the masks #########################################################
# Here we use several masks:
# * nifti_masker.mask_img_ is the original mask
# * process_mask_img is a subset of mask, it contains the voxels that should be
#   processed (we only keep the slice z = 26 and the back of the brain to speed
#   up computation)
process_mask = nifti_masker.mask_img_.get_data().astype(np.int)
process_mask[..., 38:] = 0
process_mask[..., :36] = 0
process_mask[:, 30:] = 0
process_mask_img = Nifti1Image(process_mask,
                               nifti_masker.mask_img_.get_affine())

### Searchlight ###############################################################

# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 7

### Define the score function used to evaluate classifiers
# Here we use precision which mesures proportion of true positives among
# all positives results for one class.
from sklearn.metrics import precision_score
score_func = precision_score

### Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.cross_validation import KFold
cv = KFold(y.size, k=4)

import nisl.searchlight
# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nisl.searchlight.SearchLight(nifti_masker.mask_img_,
                                      process_mask_img=process_mask_img,
                                      radius=1.5 * 3.75, n_jobs=n_jobs,
                                      score_func=score_func, verbose=1, cv=cv)
searchlight.fit(niimg, y)

### F-scores ##################################################################
from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(X_masked, y)
p_values = -np.log10(p_values)
p_values[np.isnan(p_values)] = 0
p_values[p_values > 10] = 10
p_unmasked = nifti_masker.inverse_transform(p_values).get_data()


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

### Show the F_score
pl.figure(2)
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
pl.imshow(np.rot90(mean_img[..., 37]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.imshow(np.rot90(p_ma[..., 37]), interpolation='nearest',
          cmap=pl.cm.hot)
pl.title('F-scores')
pl.axis('off')
pl.show()
