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
from scipy import stats
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
# we only keep the slice z = 26 to speed up computation
process_mask[..., 27:] = 0
process_mask[..., :26] = 0
process_mask_img = nibabel.Nifti1Image(process_mask, mask_img.get_affine())
nifti_masker = NiftiMasker(mask=process_mask_img, sessions=session,
                           memory='nilearn_cache', memory_level=1)
fmri_masked = nifti_masker.fit_transform(fmri_img)

### Perform massively univariate analysis with permuted OLS ###################
neg_log_pvals, all_scores, _, params = permuted_ols(
    conditions_encoded.reshape((-1, 1)), fmri_masked.T,
    session.reshape((-1, 1)), n_perm=10000, sparsity_threshold=.5)
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    np.ravel(neg_log_pvals)).get_data()

### Visualization #############################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Use the fmri mean image as a surrogate of anatomical data
mean_fmri = fmri_img.get_data().mean(axis=-1)

# Various plotting parameters
vmin = 10 ** -0.1  # 10% corrected
vmax = np.amax(neg_log_pvals)
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")

# Plot thresholded F-scores map
ax = grid[0]
pvals_bonferroni = 1 - stats.f.cdf(
    all_scores['score'][all_scores['iter_id'] == 0],
    1, conditions_encoded.shape[0] - params['lost_dof'] - 1)
pvals_bonferroni *= fmri_masked.shape[1]
pvals_bonferroni[pvals_bonferroni > 1.] = 1.
neg_log_pvals_bonferroni = np.zeros(neg_log_pvals.size)
orig_scores_idx = all_scores['y_id'][all_scores['iter_id'] == 0]
neg_log_pvals_bonferroni[orig_scores_idx] = - np.log10(pvals_bonferroni)
neg_log_pvals_bonferroni_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_bonferroni).get_data()
# threshold at 10% corrected
process_mask_bonf = process_mask.copy()
process_mask_bonf[neg_log_pvals_bonferroni_unmasked < vmin] = 0
p_ma = np.ma.array(neg_log_pvals_bonferroni_unmasked,
                   mask=np.logical_not(process_mask_bonf))
ax.imshow(np.rot90(mean_fmri[..., 26]), interpolation='nearest',
          cmap=plt.cm.gray)
ax.imshow(np.rot90(p_ma[..., 26]), interpolation='nearest',
          cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title('Negative log p-values\n(Bonferroni)')
ax.axis('off')

# Plot permutation p-values map
ax = grid[1]
process_mask[neg_log_pvals_unmasked < vmin] = 0
p_ma = np.ma.array(neg_log_pvals_unmasked,
                   mask=np.logical_not(process_mask))
ax.imshow(np.rot90(mean_fmri[..., 26]), interpolation='nearest',
          cmap=plt.cm.gray)
im = ax.imshow(np.rot90(p_ma[..., 26]), interpolation='nearest',
               cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title('Negative log p-values\n(permutations)')
ax.axis('off')

grid[0].cax.colorbar(im)
plt.draw()
plt.show()
