"""
Massively univariate analysis of face vs house recognition (2)
==============================================================

A permuted Ordinary Least Squares algorithm is run at each voxel in
order to detemine whether or not it shows a different mean value under a
"face viewing" condition and a "house viewing" condition.

Randomized Parcellation Based Inference [1] is also used on the same data.
It yields a better recovery of the activations as the method is almost
equivalent to applying anisotropic smoothing to the data.

"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Feb. 2014
import numpy as np
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.mass_univariate import randomized_parcellation_based_inference

### Load Haxby dataset ########################################################
dataset_files = datasets.fetch_haxby_simple()

### Mask data #################################################################
mask_img = nibabel.load(dataset_files.mask)
nifti_masker = NiftiMasker(
    standardize=True,  # important for RPBI not to be biased by anatomy
    mask=dataset_files.mask,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(dataset_files.func)

### Restrict to faces and houses ##############################################
conditions_encoded, sessions = np.loadtxt(
    dataset_files.session_target).astype("int").T
conditions = np.recfromtxt(dataset_files.conditions_target)['f0']
condition_mask = np.logical_or(conditions == 'face', conditions == 'house')
conditions_encoded = conditions_encoded[condition_mask]
fmri_masked = fmri_masked[condition_mask]

n_sessions = np.unique(sessions).size
grouped_fmri_masked = np.empty((2 * n_sessions,  # two conditions per session
                                fmri_masked.shape[1]))
grouped_conditions_encoded = np.empty((2 * n_sessions, 1))
for s in range(n_sessions):
    session_mask = sessions[condition_mask] == s
    session_house_mask = np.logical_and(session_mask,
                                        conditions[condition_mask] == 'house')
    session_face_mask = np.logical_and(session_mask,
                                       conditions[condition_mask] == 'face')
    grouped_fmri_masked[2 * s] = fmri_masked[session_house_mask].mean(0)
    grouped_fmri_masked[2 * s + 1] = fmri_masked[session_face_mask].mean(0)
    grouped_conditions_encoded[2 * s] = conditions_encoded[
        session_house_mask][0]
    grouped_conditions_encoded[2 * s + 1] = conditions_encoded[
        session_face_mask][0]

### Perform massively univariate analysis with permuted OLS ###################
# We use a two-sided t-test to compute p-values, but we keep trace of the
# effect sign to add it back at the end and thus observe the signed effect
neg_log_pvals, t_scores_original_data, _ = permuted_ols(
    grouped_conditions_encoded, grouped_fmri_masked,
    # + intercept as a covariate by default
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    two_sided_test=False,  # RPBI does not perform a two-sided test
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()

### Randomized Parcellation Based Inference ###################################
neg_log_pvals_rpbi, _, _ = randomized_parcellation_based_inference(
    grouped_conditions_encoded, grouped_fmri_masked,
    # + intercept as a covariate by default
    np.asarray(mask_img.get_data()).astype(bool),
    n_parcellations=30,  # 30 for the sake of time, 100 is recommended
    n_parcels=1000,
    threshold='auto',
    n_perm=5000,  # 5,000 for the sake of time. 10,000 is recommended
    random_state=0, memory='nilearn_cache',
    n_jobs=1, verbose=True)
neg_log_pvals_rpbi_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_rpbi).get_data()

### Visualization #############################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Use the fMRI mean image as a surrogate of anatomical data
from nilearn import image
mean_fmri = image.mean_img(dataset_files.func).get_data()

# Various plotting parameters
picked_slice = 27  # plotted slice
vmin = -np.log10(0.1)  # 10% corrected
vmax = min(np.amax(neg_log_pvals), np.amax(neg_log_pvals_rpbi))
grid = ImageGrid(plt.figure(), 111, nrows_ncols=(1, 2), direction="row",
                 axes_pad=0.05, add_all=True, label_mode="1",
                 share_all=True, cbar_location="right", cbar_mode="single",
                 cbar_size="7%", cbar_pad="1%")

# Plot permutation p-values map
ax = grid[0]
p_ma = np.ma.masked_inside(neg_log_pvals_unmasked, -vmin, vmin)[..., 0]
ax.imshow(np.rot90(mean_fmri[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.gray)
im = ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
               cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values'
             '\n(Non-parametric two-sided test'
             '\n+ max-type correction)'
             '\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

# Plot RPBI p-values map
ax = grid[1]
p_ma = np.ma.masked_less(neg_log_pvals_rpbi_unmasked[..., 0], vmin)
ax.imshow(np.rot90(mean_fmri[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.gray)
ax.imshow(np.rot90(p_ma[..., picked_slice]), interpolation='nearest',
          cmap=plt.cm.RdBu_r, vmin=-vmax, vmax=vmax)
ax.set_title(r'Negative $\log_{10}$ p-values' + '\n(RPBI)'
             + '\n\n%d detections' % (~p_ma.mask[..., picked_slice]).sum())
ax.axis('off')

# plot colorbar
colorbar = grid[1].cax.colorbar(im)
plt.subplots_adjust(0., 0.03, 1., 0.83)
plt.show()
