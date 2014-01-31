"""
Massively univariate analysis of face vs house recognition
==========================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to detemine whether or not it behaves differently under a "face
viewing" condition and a "house viewing" condition. Session number is
included as a covariable.
In order to reduce computation time required for the example, only the
three last sessions are included and only z slice number 36 is considered.

"""
import numpy as np
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

### Load Haxby dataset ########################################################
dataset_files = datasets.fetch_haxby_simple()

fmri_img = nibabel.load(dataset_files.func)
conditions_encoded, session = np.loadtxt(
    dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']

### Restrict to faces and houses ##############################################
condition_mask = np.logical_and(
    np.logical_or(conditions == 'face', conditions == 'house'),
    session > 8)
fmri_img = nibabel.Nifti1Image(fmri_img.get_data()[..., condition_mask],
                               fmri_img.get_affine().copy())
conditions_encoded = conditions_encoded[condition_mask]
session = session[condition_mask]

### Mask data #################################################################
mask_img = nibabel.load(dataset_files.mask)
process_mask = mask_img.get_data().astype(np.int)
# we only keep the slice z = 36 to speed up computation
process_mask[..., 37:] = 0
process_mask[..., :36] = 0
process_mask_img = nibabel.Nifti1Image(process_mask, mask_img.get_affine())
nifti_masker = NiftiMasker(mask=process_mask_img, sessions=session,
                           memory='nilearn_cache', memory_level=1)
fmri_masked = nifti_masker.fit_transform(fmri_img)

### Perform massively univariate analysis with permuted OLS ###################
neg_log_pvals, _, _, _ = permuted_ols(
    conditions_encoded.reshape((-1, 1)), fmri_masked.T,
     session.reshape((-1, 1)), n_perm=10000, sparsity_threshold=0.5)
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()

### Visualization #############################################################
import matplotlib.pyplot as plt

# Use the fmri mean image as a surrogate of anatomical data
mean_fmri = fmri_img.get_data().mean(axis=-1)

plt.figure()
process_mask[neg_log_pvals_unmasked[..., 0] < 1.] = 0
p_ma = np.ma.array(neg_log_pvals_unmasked,
                   mask=np.logical_not(process_mask))
plt.imshow(np.rot90(mean_fmri[..., 36]), interpolation='nearest',
           cmap=plt.cm.gray)
plt.imshow(np.rot90(p_ma[..., 36, 0]), interpolation='nearest',
           cmap=plt.cm.autumn)
plt.title('Negative log p-values')
plt.axis('off')
plt.show()
