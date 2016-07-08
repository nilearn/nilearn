"""
Carefully defining the confounds
================================

This example shows the importance of confounds removal for functional
connectivity.

"""
########################################
# We use one ADHD subject.
from nilearn import datasets
adhd = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd.func[0]

######################################
# Friston 24 motion parameters
# -------------------------------------
# They include the 6 motion parameters of the current volume
# and the preceding volume, plus each of these values squared.

# Read the 6 motion parameters from the confounds file
import numpy as np
csv_confounds = np.recfromcsv(adhd.confounds[0], delimiter='\t')
current_volume_motion = [
    csv_confounds[name] for name in ('motionx', 'motiony', 'motionz',
                                     'motionpitch', 'motionroll', 'motionyaw')]
current_volume_motion = np.array(current_volume_motion).T
previous_volume_motion = np.vstack((current_volume_motion[0],
                                    current_volume_motion[:-1]))
confounds = np.hstack((current_volume_motion, previous_volume_motion,
                       current_volume_motion ** 2,
                       previous_volume_motion ** 2))

##########################################################################
# Anatomical CompCor
# ----------------------------------------------
# Signals within noisy ROIs, such as WM and CSF, are assumed to reflect noise.
# The anatomical CompCor approach removes the first principal components
# of noisy ROIs.
#
# First, we define the tissue masks.
# As we do not have an anatomical image, we are relying on group-level masks.
# Note that subject-specific masks would give more accurate results.

##########################################################################
# We give the locations of probabilistic group masks
icbm152 = datasets.fetch_icbm152_2009()
mask_images = {'gm': icbm152.gm, 'wm': icbm152.wm, 'csf': icbm152.csf}

##########################################################################
# and use them to compute binary masks.
from nilearn import image
for tissue_name, threshold in zip(['wm', 'gm', 'csf'], ['.9', '.5', '.3']):
    mask_images[tissue_name] = image.math_img('img > ' + threshold,
                                              img=mask_images[tissue_name])

##########################################################################
# We erode WM and CSF masks, to avoid including any signal of neuronal origin.
from scipy import ndimage
for tissue_name in ['wm', 'csf']:
    mask_img = mask_images[tissue_name]
    eroded_mask_data = mask_img.get_data()
    for n_erosions in range(2):
        eroded_mask_data = ndimage.binary_erosion(eroded_mask_data)
    mask_images[tissue_name] = image.new_img_like(mask_img, eroded_mask_data)

##########################################################################
# Next, we resample the tissue masks to the functional image
from nilearn import image, masking
mean_func_img = image.mean_img(func_filename)
func_mask_img = masking.compute_epi_mask(mean_func_img)
affine = mean_func_img.get_affine()
shape = mean_func_img.get_data().shape
for tissue_name, mask_img in mask_images.items():
    mask_images[tissue_name] = image.resample_img(mask_img, target_shape=shape,
                                                  target_affine=affine,
                                                  interpolation='nearest')

##########################################################################
# and intersect them with a brain mask computed from the mean functional image.
from nilearn import masking
func_mask_img = masking.compute_epi_mask(mean_func_img, opening=0)
for tissue_name, mask_img in mask_images.items():
    mask_images[tissue_name] = image.math_img("img1 * img2", img1=mask_img,
                                              img2=func_mask_img)

##########################################################################
# We check the masks quality by plotting their contours
# on top of the mean functional image.
from nilearn import plotting
cut_coords = (15, 8, 32)  # corpus callosum
display = plotting.plot_anat(mean_func_img, cut_coords=cut_coords)
for tissue_name, colors in zip(['gm', 'wm', 'csf'], 'ymg'):
    display.add_contours(mask_images[tissue_name], colors=colors)

##########################################################################
# Now we are ready to compute the voxel-wise signals within the
# WM and CSF masks, and perform a PCA to extract the top 5 components.
# All is done by the function **nilearn.image.high_variance_confounds**,
# with the parameter **percentile** set to **100.**, to include all the voxels
# of the mask.
import numpy as np
for tissue_name in ['wm', 'csf']:
    tissue_confounds = image.high_variance_confounds(
        func_filename, mask_img=mask_images[tissue_name], n_confounds=5,
        percentile=100.)
    confounds = np.hstack((confounds, tissue_confounds))

#######################################################
# Exploring confounds contribution to GM voxels signals
# -----------------------------------------------------
# Typically motion has a dramatic impact on voxels within the contour of the
# brain.

#######################################################################
# We compute GM voxels signals
from nilearn import input_data
gm_masker = input_data.NiftiMasker(mask_img=mask_images['gm'],
                                   memory='nilearn_cache',
                                   standardize=True)
gm_signals = gm_masker.fit_transform(func_filename)

#######################################################################
# and predict voxels signals from confounds.
from sklearn import linear_model
regr = linear_model.LinearRegression(normalize=True)
regr.fit(confounds, gm_signals)
total_variance = np.sum(gm_signals ** 2, axis=0)
residual_variance = np.sum((regr.predict(confounds) - gm_signals) ** 2,
                           axis=0)
variance_percent = 100. - 100. * residual_variance / total_variance
# Transform from array to image
variance_img = gm_masker.inverse_transform(variance_percent)

#######################################################################
# The variance map shows the percent of variance explained by confounds.
display = plotting.plot_stat_map(variance_img, bg_img=mean_func_img,
                                 cut_coords=cut_coords, vmax=100.)

############################################################################
# Visualizing the impact of confounds removal on voxel-to-voxel connectivity
# --------------------------------------------------------------------------
# After confounds removal, the distribution of connectivity values should be
# tight and approximately centered on zero.

#######################################################################
# We use the same masker, but include confounds.
cleaned_gm_signals = gm_masker.fit_transform(func_filename,
                                             confounds=confounds)

#######################################################################
# We plot the histogram of voxel-to-voxel correlations for a selection of
# voxels.
import matplotlib.pylab as plt
selected_voxels = range(0, gm_signals.shape[1], 10)
plt.figure(figsize=(8, 4))
for signals, label, color in zip([gm_signals, cleaned_gm_signals],
                                 ['with confounds', 'without confounds'],
                                 'rb'):
    selected_signals = signals[:, selected_voxels]
    correlations = np.dot(selected_signals.T, selected_signals) / \
        selected_signals.shape[0]
    plt.hist(correlations[np.triu_indices_from(correlations, k=1)],
             color=color, alpha=.3, bins=100, lw=0, label=label)
plt.legend()
plt.xlabel('correlation values')
plt.show()
