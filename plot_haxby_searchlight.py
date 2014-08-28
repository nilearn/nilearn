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
from nilearn import datasets

dataset_files = datasets.fetch_haxby_simple()

fmri_img = nibabel.load(dataset_files.func)
y, session = np.loadtxt(dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']

### Restrict to faces and houses ##############################################
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')

fmri_img = nibabel.Nifti1Image(fmri_img.get_data()[..., condition_mask],
                               fmri_img.get_affine().copy())
y, session = y[condition_mask], session[condition_mask]
conditions = conditions[condition_mask]

### Prepare masks #############################################################
# - mask_img is the original mask
# - process_mask_img is a subset of mask_img, it contains the voxels that
#   should be processed (we only keep the slice z = 26 and the back of the
#   brain to speed up computation)

mask_img = nibabel.load(dataset_files.mask)

# .astype() makes a copy.
process_mask = mask_img.get_data().astype(np.int)
picked_slice = 27
process_mask[..., (picked_slice + 1):] = 0
process_mask[..., :picked_slice] = 0
process_mask[:, 30:] = 0
process_mask_img = nibabel.Nifti1Image(process_mask, mask_img.get_affine())

### Searchlight computation ###################################################

# Make processing parallel
# /!\ As each thread will print its progress, n_jobs > 1 could mess up the
#     information output.
n_jobs = 1

### Define the cross-validation scheme used for validation.
# Here we use a KFold cross-validation on the session, which corresponds to
# splitting the samples in 4 folds and make 4 runs using each fold as a test
# set once and the others as learning sets
from sklearn.cross_validation import KFold
cv = KFold(y.size, n_folds=4)

import nilearn.decoding
# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = nilearn.decoding.SearchLight(mask_img,
                                      process_mask_img=process_mask_img,
                                      radius=5.6, n_jobs=n_jobs,
                                      verbose=1, cv=cv)
searchlight.fit(fmri_img, y)

### F-scores computation ######################################################
from nilearn.input_data import NiftiMasker

# For decoding, standardizing is often very important
nifti_masker = NiftiMasker(mask=mask_img, sessions=session,
                           standardize=True, memory='nilearn_cache',
                           memory_level=1)
fmri_masked = nifti_masker.fit_transform(fmri_img)

from sklearn.feature_selection import f_classif
f_values, p_values = f_classif(fmri_masked, y)
p_values = -np.log10(p_values)
p_values[p_values > 10] = 10
p_unmasked = nifti_masker.inverse_transform(p_values).get_data()

### Visualization #############################################################
import matplotlib.pyplot as plt

# Use the fmri mean image as a surrogate of anatomical data
from nilearn import image
from nilearn.plotting import plot_stat_map
mean_fmri = image.mean_img(fmri_img)

plot_stat_map(nibabel.Nifti1Image(searchlight.scores_,
                                  mean_fmri.get_affine()), mean_fmri,
              title="Searchlight", display_mode="z", cut_coords=[-16],
              colorbar=False)

### F_score results
p_ma = np.ma.array(p_unmasked, mask=np.logical_not(process_mask))
plot_stat_map(nibabel.Nifti1Image(p_ma,
                                  mean_fmri.get_affine()), mean_fmri,
              title="F-scores", display_mode="z", cut_coords=[-16],
              colorbar=False)

plt.show()
