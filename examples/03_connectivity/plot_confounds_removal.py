"""
Careful confounds removal for functional connectivity analysis
==============================================================

This example shows the importance of confounds removal for functional
connectivity. It provides tips about the contribution of confounds to noisy
gray matter (GM) voxels and their impact on the voxel-to-voxel connectivity
distribution.

Non-neural fluctuations such as scanner instabilities, subject motion,
and physiological artifacts (e.g. respiration and cardiac effects)
disrupt functional connectivity results. As there isn't a concensus on which
nuisance signals should be considered, it is important to explore the relevance
of the chosen counfounds for the data at hand.

In this example, 24 movement regressors [1] are used to estimate head motion.
Physiological noise is assessed with the application of anatomical CompCor [2]:
10 significant principal components are derived from white matter (WM) and
cerebrospinal fluid (CSF) regions in which the time-series data are unlikely
to be modulated by neural activity. This confounds model is compared to a less
careful model with the same number of confounds.

References
----------
[1] Friston, K. J. et al. (1996).
    Movement-related effects in fMRI time-series. Magnetic Resonance in
    Medicine, 35(3), 346-355.

[2] Behzadi, Y. et al. (2007).
    A Component Based Noise Correction Method (CompCor) for BOLD and Perfusion
    Based fMRI. Neuroimage, 37(1), 90-101.
"""
######################################
# Getting the data
# -----------------------------------------
# We use single subject data from nilearn provided ADHD datasets.
from nilearn import datasets
adhd = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd.func[0]

##########################################################################
# Data include the functional image and a csv file of possible confounds.
print(func_filename)
print(adhd.confounds[0])

######################################
# Friston et al. 24 motion parameters model
# -----------------------------------------
# They include the 6 motion parameters of the current volume and the preceding
# volume, plus each of these values squared.

# Read the 6 motion parameters from the confounds file
import numpy as np
csv_confounds = np.recfromcsv(adhd.confounds[0], delimiter='\t',
                              usecols=('motionx', 'motiony', 'motionz',
                                       'motionpitch', 'motionroll',
                                       'motionyaw'))
current_volume_motion = csv_confounds.view(dtype=float).reshape(-1, 6)
preceding_volume_motion = np.vstack((current_volume_motion[0],
                                     current_volume_motion[:-1]))
motion_confounds = np.hstack((current_volume_motion, preceding_volume_motion,
                              current_volume_motion ** 2,
                              preceding_volume_motion ** 2))

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
# We give the paths to probabilistic group masks.
icbm152 = datasets.fetch_icbm152_2009()
mask_images = {'GM': icbm152.gm, 'WM': icbm152.wm, 'CSF': icbm152.csf}
print("\n".join(mask_images.values()))

##########################################################################
# We can visualize these masks using `nilearn.plotting.plot_anat`.
# For this, we need to transform the images from 3D to 4D.
from nilearn import plotting
from nilearn._utils import check_niimg
for tissue_name, mask_img in mask_images.items():
    mask_img_4d = check_niimg(mask_img, atleast_4d=True)
    plotting.plot_anat(
        mask_img_4d, display_mode='z', cut_coords=1, title=tissue_name)

##########################################################################
# We use **nilearn.image.math_img** to convert probabilistic masks to binary
# masks with thresholds being adjusted manually.
from nilearn import image
for tissue_name, threshold in zip(['WM', 'GM', 'CSF'], ['.9', '.5', '.3']):
    mask_images[tissue_name] = image.math_img('img > ' + threshold,
                                              img=mask_images[tissue_name])

##########################################################################
# We erode WM and CSF masks, to avoid including any signal of neuronal origin.
from scipy import ndimage
for tissue_name in ['WM', 'CSF']:
    mask_img = mask_images[tissue_name]
    eroded_mask_data = ndimage.binary_erosion(mask_img.get_data(),
                                              iterations=2)
    mask_images[tissue_name] = image.new_img_like(mask_img, eroded_mask_data)

##########################################################################
# Next, we resample the tissue masks to the functional image
mean_func_img = image.mean_img(func_filename)
for tissue_name, mask_img in mask_images.items():
    mask_images[tissue_name] = image.resample_to_img(mask_img, mean_func_img,
                                                     interpolation='nearest')

##########################################################################
# and intersect them with a brain mask computed from the mean functional image.
from nilearn import masking
func_mask_img = masking.compute_epi_mask(mean_func_img, opening=0)
for tissue_name, mask_img in mask_images.items():
    mask_images[tissue_name] = masking.intersect_masks(
        [func_mask_img, mask_img], threshold=1, connected=False)

##########################################################################
# We check the masks quality by plotting them as contours
# on top of the mean functional image.
display = plotting.plot_anat(mean_func_img)
for tissue_name, colors in zip(['GM', 'WM', 'CSF'], 'ymg'):
    display.add_contours(mask_images[tissue_name], colors=colors)

##########################################################################
# Now we are ready to compute the voxel-wise signals within the
# WM and CSF masks, and perform a PCA to extract the top 5 components.
# All is done by the function `nilearn.image.high_variance_confounds`,
# with the parameter **percentile** set to **100.**, to include all the voxels
# of the mask.
wm_confounds = image.high_variance_confounds(
    func_filename, mask_img=mask_images['WM'], n_confounds=5, percentile=100.)
csf_confounds = image.high_variance_confounds(
    func_filename, mask_img=mask_images['CSF'], n_confounds=5, percentile=100.)

#######################################################################
# The so-computed CompCor confounds are stacked to motion confounds.
confounds1 = np.hstack((motion_confounds, wm_confounds, csf_confounds))
print('Careful model includes {0} confounds.'.format(confounds1.shape[1]))

#######################################################################
# For comparison, we compute less careful confounds of the same number
hv_confounds = image.high_variance_confounds(func_filename, n_confounds=10)
confounds2 = np.hstack((current_volume_motion, hv_confounds))
print('Gross model includes {0} confounds.'.format(confounds2.shape[1]))

#######################################################################
# Exploring confounds contribution to GM voxels signals
# -----------------------------------------------------
# Typically motion has a dramatic impact on voxels within the contour of the
# brain. Let us evaluate the confounds efficiency in modeling GM noise
# elements.

#######################################################################
# We compute GM voxels signals with
# :class:`nilearn.input_data.NiftiMasker`
from nilearn import input_data
gm_masker = input_data.NiftiMasker(mask_img=mask_images['GM'],
                                   memory='nilearn_cache',
                                   standardize=True,
                                   verbose=1)
gm_signals = gm_masker.fit_transform(func_filename)

#######################################################################
# and predict them from confounds.
from sklearn import linear_model
f_statistic_images = {}
regr = linear_model.LinearRegression(normalize=True)
for (confounds, label) in zip([confounds1, confounds2],
                              ['careful confounds', 'gross confounds']):
    regr.fit(confounds, gm_signals)
    predicted_gm_signals = regr.predict(confounds)
    n_samples, n_features = confounds.shape
    mean_square_regression = np.sum(predicted_gm_signals ** 2,
                                    axis=0) / (n_features - 1)
    mean_square_error = np.sum((predicted_gm_signals - gm_signals) ** 2,
                               axis=0) / (n_samples - n_features)
    f_statistic = mean_square_regression / mean_square_error
    # Transform from array to image
    f_statistic_images[label] = gm_masker.inverse_transform(f_statistic)

#######################################################################
# The F-map reflects the overall fit of the linear model explaining the GM
# voxel-wise time-series by the confounds.
for (title, f_statistic_img) in f_statistic_images.items():
    display = plotting.plot_stat_map(f_statistic_img,
                                     bg_img=mean_func_img,
                                     display_mode='z',
#                                     cut_coords=[-20., 20.],
#                                     vmax=100.,
                                     title=title,
#                                     cmap=plotting.cm.purple_green
                                     threshold=10.)

############################################################################
# Visualizing the impact of confounds removal on voxel-to-voxel connectivity
# --------------------------------------------------------------------------
# After confounds removal, the distribution of connectivity values should be
# tight and approximately centered to zero.

#######################################################################
# We use the same masker, but include confounds.
cleaned_gm_signals1 = gm_masker.fit_transform(func_filename,
                                              confounds=confounds1)
cleaned_gm_signals2 = gm_masker.fit_transform(func_filename,
                                              confounds=confounds2)

#######################################################################
# Finally, we compute the voxel-to-voxel Pearson'r correlations for a selection
# of voxels
from sklearn.covariance import EmpiricalCovariance
from nilearn import connectome
connectivity_measure = connectome.ConnectivityMeasure(
    cov_estimator=EmpiricalCovariance(), kind='correlation')
selected_voxels = range(0, gm_signals.shape[1], 10)
correlations = {}
labels = ['no confounds\nremoval', 'careful confounds\nremoval',
          'gross confounds\nremoval']
for (signals, label) in zip(
        [gm_signals, cleaned_gm_signals1, cleaned_gm_signals2], labels):
    correlations[label] = connectivity_measure.fit_transform(
        [signals[:, selected_voxels]])[0]

#######################################################################
# and plot their histograms.
import matplotlib.pylab as plt
plt.figure(figsize=(8, 4))
for label, color in zip(labels, 'rgb'):
    correlation_values = correlations[label]
    plt.hist(correlation_values[np.triu_indices_from(correlation_values, k=1)],
             color=color, alpha=.3, bins=100, lw=0, label=label)

plt.vlines(0, plt.ylim()[0], plt.ylim()[1])
plt.legend(loc='center left')
plt.xlabel('correlation values')
plt.show()
