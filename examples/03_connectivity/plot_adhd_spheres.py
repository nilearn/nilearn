"""
Extracting brain signal from spheres
====================================

This example extracts brain signals from spheres described by the coordinates
of their center in MNI space and a given radius in millimeters. In particular,
this example extracts signals from 4 Default Mode Network seeds (posterior
cingulate cortex, medial prefrontal cortex and left and right temporoparietal
junction) and computes inverse covariance between them. The example concludes
with a more advanced part dedicated to spheres radius choice.

"""

###############################################################################
# Loding fMRI data and giving spheres centers
# -------------------------------------------

###############################################################################
# We retrieve the first subject data of the ADHD dataset.
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: {0}'.format(
    func_filename))  # 4D data


###############################################################################
# We give coordinates of 4 regions from the Default Mode Network.
dmn_coords = [(0, -52, 18), (1, 50, -5), (-46, -68, 32), (46, -68, 32)]
labels = ['PCC', 'MPFC', 'rTPJ', 'lTPJ']

###############################################################################
# It is advised to intersect the spheric regions with a grey matter mask. Since
# we don't have subject grey mask, we resort to a less precise group-level one.
icbm152_grey_mask = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# Computing average signals on 10mm radius spheres
# ------------------------------------------------

###############################################################################
# We extract signal from sphere around DMN seeds.
from nilearn import input_data

dmn_10mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=10.,
    mask_img=icbm152_grey_mask, detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)
dmn_10mm_average_time_series = dmn_10mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

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

###############################################################################
# We can study the direct connections between the DMN ROIs by estimating
# the precision or inverse covariance matrix.
# This can be done through the Ledoit-Wolf covariance estimator, well suited
# when the number of ROIs is small compared to the number of samples.
from sklearn.covariance import LedoitWolf
estimator = LedoitWolf()


###############################################################################
# We fit the estimator with the DMN timeseries,
estimator.fit(dmn_10mm_average_time_series)

###############################################################################
# and get the direct connections strength, revealed by the precision matrix. 

# negated precision coefficients are proportional to partial correlations.
connectivity_matrix = -estimator.precision_


##########################################################################
# We can check that we got a square (n_spheres, n_spheres) connectivity matrix.
print('inverse covariance matrix has shape {0}'.format(
    connectivity_matrix.shape))

##########################################################################
# Visualizing the connections
# ---------------------------

##########################################################################
# We can display the connectivity graph with hemispheric projections using
# the connectome dedicated function `nilearn.plotting.plot_connectome`.
# Connectivity values are reflected by edges colors.
from nilearn import plotting

title = "Connectivity projected on hemispheres"
plotting.plot_connectome(connectivity_matrix, dmn_coords, title=title,
                         display_mode='lyrz')

##########################################################################
# Notice that PCC is included in both hemispheres since x == 0.

##########################################################################
# Hesitating on the radius? Examine voxel-wise timeseries within spheres
# ----------------------------------------------------------------------

##########################################################################
# There is no concensus on the spheres radius: They may range across studies
# from 4 mm to 12 mm... One way is to explore the homogenity
# of times-series within the sphere.

##########################################################################
# We start by computing the grey matter voxel-wise time series.

import numpy as np
from nilearn.image.resampling import coord_transform
gm_masker = input_data.NiftiMasker(
    mask_img=icbm152_grey_mask, standardize=True, #low_pass=0.1, high_pass=0.01,
    t_r=2.5, detrend=True, memory='nilearn_cache', memory_level=1)
gm_voxels_time_series = gm_masker.fit_transform(func_filename,
                                                confounds=[confound_filename])

###############################################################################
# Then compute voxels indices within the grey mask
gm_mask_img = gm_masker.mask_img
# data in the gray mask image is stored in a 3D array
gm_mask_array = gm_mask_img.get_data()
i, j, k = np.where(gm_mask_array != 0)

###############################################################################
# and transform indices to MNI coordiantes.
gm_mask_mni_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

###############################################################################
# We can use these coordinates to identify voxels lying inside a given seed.
# For instance for PCC
pcc_coords = dmn_coords[0]
sphere_10mm_mask = np.linalg.norm(
    gm_mask_mni_coords - pcc_coords, axis=1) <= 10.
pcc_10mm_voxels_time_series = gm_voxels_time_series[:, sphere_10mm_mask]

print('PCC seed has {0} voxels'.format(pcc_10mm_voxels_time_series.shape[1]))

###############################################################################
# We can display the time-series of 8 random voxels within the PCC sphere
plt.figure()
plt.plot(pcc_10mm_voxels_time_series[:, [3, 9, 16, 292, 531, 861, 150, 1708]])
plt.title('PCC voxel-wise time-series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')

###############################################################################
# Computing seed-to-voxels correlations within the sphere
# -------------------------------------------------------

###############################################################################
# We can compute the correlation between the average PCC time-series and PCC
# voxel-wise signals
pcc_10mm_mean_time_series = dmn_10mm_average_time_series[:, 0]
# Normalize by the scans number to get correlation of standardized signals
pcc_10mm_seed_to_voxel_correlation = np.dot(
    pcc_10mm_voxels_time_series.T,
    pcc_10mm_mean_time_series) / pcc_10mm_mean_time_series.shape[0]


###############################################################################
# and form a 3D spheric correlation map.
array = np.zeros(sphere_10mm_mask.shape)
array[sphere_10mm_mask] = pcc_10mm_seed_to_voxel_correlation
# transform the 2D array back to a 3D image 
pcc_10mm_seed_to_voxel_correlation_img = gm_masker.inverse_transform(
    array)

###############################################################################
# We can repeat the operation looping over all seeds.
dmn_10mm_to_voxel_correlation_imgs = []
print('--- 10 mm spheres: correlations range from ---')
for seed_coords, label, seed_10mm_mean_time_series in zip(
        dmn_coords, labels, dmn_10mm_average_time_series.T):
    sphere_10mm_mask = np.linalg.norm(
        gm_mask_mni_coords - seed_coords, axis=1) <= 10.
    seed_10mm_voxels_time_series = gm_voxels_time_series[:, sphere_10mm_mask]
    seed_10mm_to_voxel_correlation = np.dot(
        seed_10mm_voxels_time_series.T,
        seed_10mm_mean_time_series) / seed_10mm_mean_time_series.shape[0]
    print('   {0} to {1} for {2}'.format(
        seed_10mm_to_voxel_correlation.min(),
        seed_10mm_to_voxel_correlation.max(), label))
    array = np.zeros(sphere_10mm_mask.shape)
    array[sphere_10mm_mask] = seed_10mm_to_voxel_correlation
    dmn_10mm_to_voxel_correlation_imgs.append(
        gm_masker.inverse_transform(array))


###############################################################################
# We got a list of spheric correlation maps for each seed. We can visualize it
# on top of the connectome plot.
title = 'DMN 10mm seed-to-voxels within-seed correlation maps'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    -precision_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure)

# overlay the connectome display with the spheric correlation maps.
for seed_10mm_to_voxel_correlation_img in dmn_10mm_to_voxel_correlation_imgs:
    display.add_overlay(seed_10mm_to_voxel_correlation_img, vmax=1., vmin=-1,
                        cmap=plotting.cm.cold_hot)

###############################################################################
# Decreasing the radius
# ---------------------

###############################################################################
# Let's repeat all the previous steps with 5mm radius.
dmn_5mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=5., mask_img=icbm152_grey_mask,
    detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1)
dmn_5mm_mean_time_series = dmn_5mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

estimator = LedoitWolf()
estimator.fit(dmn_5mm_mean_time_series)
precision_matrix = estimator.precision_

dmn_5mm_to_voxel_correlation_imgs = []
print('--- 5 mm spheres: correlations range from ---')
for seed_coords, label, seed_5mm_mean_time_series in zip(
        dmn_coords, labels, dmn_5mm_mean_time_series.T):
    sphere_5mm_mask = np.linalg.norm(
        gm_mask_mni_coords - seed_coords, axis=1) <= 5.
    seed_5mm_voxels_time_series = gm_voxels_time_series[:, sphere_5mm_mask]
    seed_5mm_to_voxel_correlation = np.dot(
        seed_5mm_voxels_time_series.T,
        seed_5mm_mean_time_series) / seed_5mm_mean_time_series.shape[0]
    print('   {0} to {1} for {2}'.format(
        seed_5mm_to_voxel_correlation.min(),
        seed_5mm_to_voxel_correlation.max(), label))
    array = np.zeros(sphere_5mm_mask.shape)
    array[sphere_5mm_mask] = seed_5mm_to_voxel_correlation
    dmn_5mm_to_voxel_correlation_imgs.append(
        gm_masker.inverse_transform(array))

###############################################################################
# Correlations maps show higher correlations, and no anti-correlation.

title = 'DMN 5mm seed-to-voxels within-seed correlation maps'
figure = plt.figure(figsize=(6, 6))
display = plotting.plot_connectome(
    -precision_matrix, dmn_coords, node_size=0,
    title=title, display_mode='z', figure=figure)
for seed_5mm_to_voxel_correlation_img in dmn_5mm_to_voxel_correlation_imgs:
    display.add_overlay(seed_5mm_to_voxel_correlation_img, vmax=1., vmin=-1,
                        cmap=plotting.cm.cold_hot)

plotting.show()

###############################################################################
# We also succeed to recover better hemispheric connection, and observe
# anti-correlation between left and right regions of temporo-parietal junction.
