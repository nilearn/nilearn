"""
Extracting brain signal from spheres
====================================

This example shows how to extract brain signals from spheres described by the
coordinates of their center in MNI space and a given radius in millimeters.
In particular, it creates 4 ROIs part of the default mode network (DMN) as
10-mm-radius spheres centered at coordinates from [1] and estimates partial
correlation strength between them. The example concludes with a more advanced
part dedicated to spheres radius choice.


** References **

[1] Vincent JL, Kahn I, Snyder AZ, Raichle ME, Buckner RL (2008) Evidence
for a frontoparietal control system revealed by intrinsic functional con-
nectivity. J Neurophysiol 100:3328-3342.

"""

###############################################################################
# Loding fMRI data and giving spheres centers
# -------------------------------------------
#
# We retrieve the first subject data of the ADHD dataset.
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: {0}'.format(
    func_filename))  # 4D data


###############################################################################
# We give coordinates of the posterior cingulate cortex,  the medial prefrontal
# cortex and the left and right angular gyrus, all part of the DMN.
pcc_coords = (1, -55, 17)
mpfc_coords = (0, 51, -7)
lag_coords = (-47, -71, 29)
rag_coords = (50, -64, 27)

dmn_coords = [pcc_coords, mpfc_coords, lag_coords, rag_coords]
labels = ['PCC', 'MPFC', 'lAG', 'rAG']

###############################################################################
# It is advised to mask the spheres with grey matter. We don't have subject's
# grey matter mask, we resort to a less precise group-level one.
gm_mask_img = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# Creating the ROIs
#------------------
#
# We compute voxels indices within the grey mask
# data in the gray mask image is stored in a 3D array
import numpy as np

gm_mask_array = gm_mask_img.get_data()
i, j, k = np.where(gm_mask_array != 0)


from nilearn.image.resampling import coord_transform
gm_mask_mni_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

###############################################################################
# We use the obtained coordinates to identify voxels lying inside the seeds.
pcc_10mm_mask = np.linalg.norm(gm_mask_mni_coords - pcc_coords, axis=1) < 10.
mpfc_10mm_mask = np.linalg.norm(gm_mask_mni_coords - mpfc_coords, axis=1) < 10.
lag_10mm_mask = np.linalg.norm(gm_mask_mni_coords - lag_coords, axis=1) < 10.
rag_10mm_mask = np.linalg.norm(gm_mask_mni_coords - rag_coords, axis=1) < 10.

dmn_10mm_mask = pcc_10mm_mask + mpfc_10mm_mask + lag_10mm_mask + rag_10mm_mask

from nilearn import masking
dmn_10mm_mask_img = masking.unmask(dmn_10mm_mask, gm_mask_img)

from nilearn import plotting
plotting.plot_roi(dmn_10mm_mask_img, cut_coords=[0, -68, 28])

###############################################################################
# We can visualize 

###############################################################################
# Computing average signals on 10mm radius spheres
# ------------------------------------------------
#
# We make use of a spheric ROIs dedicated object, the
# `:class:nilearn.input_data.NiftiSpheresMasker`. We define it by specifying
# the spheres centers and radius, as well as optional arguments:
# The grey matter mask image, the detrending, signal normalization and
# filtering choices.
from nilearn import input_data

dmn_10mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=10., mask_img=gm_mask_img,
    low_pass=0.1, t_r=2.5, high_pass=None, detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=1)

###############################################################################
# Time-series are computed once functional and confounds filenames given.
dmn_10mm_average_time_series = dmn_10mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

print('Saved seeds time-series in array of shape {0}'.format(
    dmn_10mm_average_time_series.shape))

###############################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt

plt.figure()
for time_serie, label in zip(dmn_10mm_average_time_series.T, labels):
    plt.plot(time_serie, label=label, lw=3.)

plt.title('Default Mode Network Average Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()


###############################################################################
# Estimating connectivity
# -----------------------
#
# Direct connections between ROIs are revealed by the signals precision
# (ie inverse covariance) matrix. We compute this matrix using the Ledoit-Wolf
# estimator, well suited to small number of ROIs /large number of scans cases.
from sklearn.covariance import LedoitWolf

estimator = LedoitWolf()

###############################################################################
# We just need to fit the estimator with the DMN timeseries.
estimator.fit(dmn_10mm_average_time_series)

# negated precision coefficients are proportional to partial correlations.
connectivity_matrix = -estimator.precision_  # no it's not connectivity matrix

###############################################################################
# We can check that we got a square (n_spheres, n_spheres) connectivity matrix.
print('connectivity matrix has shape {0}'.format(
    connectivity_matrix.shape))

###############################################################################
# Visualizing the connections
# ---------------------------

##########################################################################
# We display the connectivity graph with hemispheric projections using
# the connectome dedicated function `nilearn.plotting.plot_connectome`.
# Connectivity values are reflected by edges colors.
from nilearn import plotting

title = "Connectivity projected on hemispheres"
plotting.plot_connectome(connectivity_matrix, dmn_coords, title=title,
                         display_mode='lyrz')


title = 'DMN 10mm seed-to-voxel correlation maps within-seeds'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    connectivity_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure, edge_vmax=.6)

display.add_overlay(dmn_10mm_mask_img)#, levels=[0, 1])
plt.text(pcc_coords[0] + 10., pcc_coords[1] + 10., 'PCC')
plt.text(mpfc_coords[0] + 10., mpfc_coords[1] + 10., 'MPFC')
plt.text(lag_coords[0] + 10., lag_coords[1] + 10., 'lAG')
plt.text(rag_coords[0] + 10., rag_coords[1] + 10., 'rAG')

plotting.show()

stop
for label, seed_coords, seed_10mm_to_voxel_correlation_img in zip(
        labels, dmn_coords, dmn_10mm_mask_imgs):
    # Overlay the connectome display with the correlation map
    display.add_overlay(seed_10mm_to_voxel_correlation_img, vmax=1., vmin=-1,
                        cmap=plotting.cm.cold_hot)
    plt.text(seed_coords[0] + 10., seed_coords[1] + 10., label)


##########################################################################
# Notice that MPCC is included in both hemispheres since x == 0.

##########################################################################
# Hesitating on the radius? Examine voxel-wise timeseries within spheres
# ----------------------------------------------------------------------
#
# There is no concensus on the spheres radius: They range from 4 to 12 mm
# across studies. Exploring the times-series within the spheres is helpful.

##########################################################################
# We start by computing the grey matter voxel-wise time series.
import numpy as np
from nilearn.image.resampling import coord_transform

gm_masker = input_data.NiftiMasker(
    mask_img=gm_mask_img, standardize=True,
    low_pass=0.1, t_r=2.5, high_pass=None, detrend=True,
    memory='nilearn_cache', memory_level=1, verbose=2)

gm_voxels_time_series = gm_masker.fit_transform(func_filename,
                                                confounds=[confound_filename])
print('Gray matter voxel-wise signals stored in array of shape {0}'.format(
    gm_voxels_time_series.shape))

###############################################################################
# Then we compute voxels indices within the grey mask
gm_mask_img = gm_masker.mask_img
# data in the gray mask image is stored in a 3D array
gm_mask_array = gm_mask_img.get_data()
i, j, k = np.where(gm_mask_array != 0)

###############################################################################
# and transform all these indices to MNI coordiantes.
gm_mask_mni_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

###############################################################################
# We can use the obtained coordinates to identify voxels lying inside a given
# seed. For instance for the PCC
pcc_coords = dmn_coords[0]
pcc_10mm_mask = np.linalg.norm(
    gm_mask_mni_coords - pcc_coords, axis=1) < 10.
pcc_10mm_voxels_time_series = gm_voxels_time_series[:, pcc_10mm_mask]

print('PCC seed has {0} voxels'.format(pcc_10mm_voxels_time_series.shape[1]))

###############################################################################
# We can display the time-series of 8 random voxels within the PCC sphere.
plt.figure()
plt.plot(pcc_10mm_voxels_time_series[:, [3, 9, 16, 292, 531, 861, 150, 1708]])
plt.title('PCC voxel-wise time-series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')

###############################################################################
# Computing seed-to-voxels correlations within the sphere
# -------------------------------------------------------
#
# We compute the correlation between the mean PCC signal and the within PCC
# voxel-wise time-series.
pcc_10mm_mean_time_series = dmn_10mm_average_time_series[:, 0]
# Normalize by the scans number to get correlation of standardized signals
pcc_10mm_seed_to_voxel_correlation = np.dot(
    pcc_10mm_voxels_time_series.T,
    pcc_10mm_mean_time_series) / pcc_10mm_mean_time_series.shape[0]

print('correlations within PCC stored in {0} array'.format(
    pcc_10mm_seed_to_voxel_correlation.shape))

###############################################################################
# We form a seed-to-voxel correlation map restricted to the PCC voxels. First
# we fill the `pcc_10mm_mask` array with the correlations values.
pcc_10mm_filled_mask = np.array(pcc_10mm_mask, dtype=float)
pcc_10mm_filled_mask[pcc_10mm_mask] = pcc_10mm_seed_to_voxel_correlation

###############################################################################
# Then we transform the 2D array back to a 3D image using `gm_masker`
pcc_10mm_seed_to_voxel_correlation_img = gm_masker.inverse_transform(
    pcc_10mm_filled_mask)

###############################################################################
# We can repeat the previous steps looping over all seeds.
dmn_10mm_to_voxel_correlation_imgs = []
print('--- 10 mm spheres: correlations range from ---')
for seed_coords, label, seed_10mm_mean_time_series in zip(
        dmn_coords, labels, dmn_10mm_average_time_series.T):
    seed_10mm_mask = np.linalg.norm(
        gm_mask_mni_coords - seed_coords, axis=1) <= 10.
    seed_10mm_voxels_time_series = gm_voxels_time_series[:, seed_10mm_mask]
    seed_10mm_to_voxel_correlation = np.dot(
        seed_10mm_voxels_time_series.T,
        seed_10mm_mean_time_series) / seed_10mm_mean_time_series.shape[0]
    print('   {0:1.2} to {1:1.2} for {2}'.format(
        seed_10mm_to_voxel_correlation.min(),
        seed_10mm_to_voxel_correlation.max(), label))
    seed_10mm_filled_mask = np.array(seed_10mm_mask, dtype=float)
    seed_10mm_filled_mask[seed_10mm_mask] = seed_10mm_to_voxel_correlation
    dmn_10mm_to_voxel_correlation_imgs.append(
        gm_masker.inverse_transform(seed_10mm_filled_mask))


###############################################################################
# We got a list of spheric correlation maps for each seed. We can visualize
# them on top of the connectome plot.
title = 'DMN 10mm seed-to-voxel correlation maps within-seeds'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    connectivity_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure, edge_vmax=.6)

for label, seed_coords, seed_10mm_to_voxel_correlation_img in zip(
        labels, dmn_coords, dmn_10mm_to_voxel_correlation_imgs):
    # Overlay the connectome display with the correlation map
    display.add_overlay(seed_10mm_to_voxel_correlation_img, vmax=1., vmin=-1,
                        cmap=plotting.cm.cold_hot)
    plt.text(seed_coords[0] + 10., seed_coords[1] + 10., label)

###############################################################################
# Decreasing the radius
# ---------------------
#
# Let's repeat the whole procedure with 5mm radius.
dmn_5mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=5., mask_img=gm_mask_img,
    low_pass=0.1, t_r=2.5, high_pass=None,
    detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1)
dmn_5mm_mean_time_series = dmn_5mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

estimator = LedoitWolf()
estimator.fit(dmn_5mm_mean_time_series)
connectivity_matrix = -estimator.precision_

dmn_5mm_to_voxel_correlation_imgs = []
print('--- 5 mm spheres: correlations range from ---')
for seed_coords, label, seed_5mm_mean_time_series in zip(
        dmn_coords, labels, dmn_5mm_mean_time_series.T):
    seed_5mm_mask = np.linalg.norm(
        gm_mask_mni_coords - seed_coords, axis=1) <= 5.
    seed_5mm_voxels_time_series = gm_voxels_time_series[:, seed_5mm_mask]
    seed_5mm_to_voxel_correlation = np.dot(
        seed_5mm_voxels_time_series.T,
        seed_5mm_mean_time_series) / seed_5mm_mean_time_series.shape[0]
    print('   {0:1.2} to {1:1.2} for {2}'.format(
        seed_5mm_to_voxel_correlation.min(),
        seed_5mm_to_voxel_correlation.max(), label))
    seed_5mm_filled_mask = np.zeros(seed_5mm_mask.shape)
    seed_5mm_filled_mask[seed_5mm_mask] = seed_5mm_to_voxel_correlation
    dmn_5mm_to_voxel_correlation_imgs.append(
        gm_masker.inverse_transform(seed_5mm_filled_mask))


###############################################################################
# Correlation maps show higher correlations, and no anti-correlation.

title = 'DMN 5mm seed-to-voxel correlation maps within-seeds'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    connectivity_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure, edge_vmax=.6)
for label, seed_coords, seed_5mm_to_voxel_correlation_img in zip(
        labels, dmn_coords, dmn_5mm_to_voxel_correlation_imgs):
    display.add_overlay(seed_5mm_to_voxel_correlation_img, vmax=1., vmin=-1,
                        cmap=plotting.cm.cold_hot)
    plt.text(seed_coords[0] + 5., seed_coords[1] + 5., label)

plotting.show()

###############################################################################
# ROI-to-ROI connections slightly change.
