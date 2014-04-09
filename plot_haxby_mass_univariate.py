"""
Massively univariate analysis of face vs house recognition
==========================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to detemine whether or not it behaves differently under a "face
viewing" condition and a "house viewing" condition.
In order to reduce computation time required for the example, only one
brain slice is considered.

The example shows the small differences that exist between
Bonferroni-corrected p-values and family-wise corrected p-values obtained
from a permutation test combined with a max-type procedure.
Bonferroni correction is a bit conservative, as revealed by the presence of
a few false negative.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

### Load Haxby dataset ########################################################
dataset_files = datasets.fetch_haxby_simple()

### Mask data #################################################################
mask_img = nibabel.load(dataset_files.mask)
nifti_masker = NiftiMasker(
    mask=dataset_files.mask,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(dataset_files.func)

### Restrict to faces and houses ##############################################
conditions_encoded, _ = np.loadtxt(
    dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
conditions_encoded = conditions_encoded[condition_mask]
fmri_masked = fmri_masked[condition_mask]

### Perform massively univariate analysis with permuted OLS ###################
neg_log_pvals, all_scores, _ = permuted_ols(
    conditions_encoded, fmri_masked,  # + intercept as a covariate by default
    n_perm=10000,
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()

### scikit-learn F-scores for comparison ######################################
from nilearn._utils.fixes import f_regression
_, pvals_bonferroni = f_regression(
    fmri_masked, conditions_encoded)  # f_regression implicitly adds intercept
pvals_bonferroni *= fmri_masked.shape[1]
pvals_bonferroni[np.isnan(pvals_bonferroni)] = 1
pvals_bonferroni[pvals_bonferroni > 1] = 1
neg_log_pvals_bonferroni = -np.log10(pvals_bonferroni)
neg_log_pvals_bonferroni_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_bonferroni).get_data()

### Visualization #############################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Use the fmri mean image as a surrogate of anatomical data
from nilearn import image
mean_fmri = image.mean_img(dataset_files.func).get_data()

# Various plotting parameters
picked_slice = 27  # plotted slice
vmin = -np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals), np.amax(neg_log_pvals_bonferroni))
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")

# Plot thresholded F-scores map
ax = grid[0]
p_ma = np.ma.masked_less(neg_log_pvals_bonferroni_unmasked, vmin)
ax.imshow(np.rot90(mean_fmri[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.gray)
ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values' + '\n(Parametric F-test + '
             '\nBonferroni correction)' +
             '\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

# Plot permutation p-values map
ax = grid[1]
p_ma = np.ma.masked_less(neg_log_pvals_unmasked, vmin)[..., 0]
ax.imshow(np.rot90(mean_fmri[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.gray)
im = ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
               cmap=plt.cm.autumn, vmin=vmin, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values' + '\n(Non-parametric + '
             '\nmax-type correction)' +
             '\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

grid[0].cax.colorbar(im)
plt.subplots_adjust(0., 0.03, 1., 0.83)
plt.show()
