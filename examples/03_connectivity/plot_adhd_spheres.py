"""
Extracting brain signal from spheres
====================================

This example extracts brain signals from spheres described by the coordinates
of their center in MNI space and a given radius in millimeters. In particular,
this example extracts signals from Default Mode Network regions and computes
inverse covariance between them.

"""

###############################################################################
# Data preparation
# ----------------

###############################################################################
# We retrieve the first subject data of the ADHD dataset.
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


###############################################################################
# We manually give coordinates of 4 regions from the Default Mode Network.
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
    'Posterior Cingulate Cortex',
    'Left Temporoparietal junction',
    'Right Temporoparietal junction',
    'Medial prefrontal cortex'
]

###############################################################################
# Within spheres average signal extraction
# ----------------------------------------

###############################################################################
# It is advised to intersect the spheric regions with a grey matter mask.
# Since we do not have subject grey mask, we resort to a less precise
# group-level one.
icbm152_grey_mask = datasets.fetch_icbm152_brain_gm_mask()

###############################################################################
# We extracts signal from sphere around DMN seeds.
from nilearn import input_data

dmn_8mm_masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8.,
    mask_img=icbm152_grey_mask, detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

dmn_8mm_average_time_series = dmn_8mm_masker.fit_transform(
    func_filename, confounds=[confound_filename])

###############################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt
plt.figure()
for time_serie, label in zip(dmn_8mm_average_time_series.T, labels):
    plt.plot(time_serie, label=label, lw=3.)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Preprocessed signal')
plt.legend()
plt.tight_layout()


###############################################################################
# Connectivity estimation
# -----------------------

###############################################################################
# We can study the conditional dependence between the DMN spheres by
# estimating the associated precision or inverse covariance matrix.
# This can be done through the Ledoit-Wolf covariance estimator, well suited
# when the number of ROIs is small compared to the number of samples.
from sklearn.covariance import LedoitWolf
estimator = LedoitWolf()


##########################################################################
# We fit the estimator with the DMN timeseries, and output the presion matrix
estimator.fit(dmn_8mm_average_time_series)
precision_matrix = estimator.precision_

##########################################################################
# We can check that we got a square (n_spheres, n_spheres) connectivity matrix.
print('inverse covariance matrix has shape {0}'.format(
    precision_matrix.shape))

##########################################################################
# Visualizing the connections
# ---------------------------

##########################################################################
# Negated precision coefficients are proportional to partial correlations,
# revealing direct connections strength. We can display them with
# the connectome dedicated function `nilearn.plotting.plot_connectome`.
# Connectivity values are reflected by edges colors.
from nilearn import plotting

# Tweak edges linewidth, for better visualization
plotting.plot_connectome(-precision_matrix, dmn_coords, edge_kwargs={'lw': 4.},
                         title="Default Mode Network Connectivity")

###############################################################################
# We can display connectome with hemispheric projections.
# Notice (0, -52, 18) is included in both hemispheres since x == 0.
title = "Connectivity projected on hemispheres"
plotting.plot_connectome(-precision_matrix, dmn_coords, title=title,
                         edge_kwargs={'lw': 4.},
                         display_mode='lyrz')

##########################################################################
# Hesitating on the radius? Examin within spheres timeseries
# ----------------------------------------------------------

##########################################################################
# There is no concensus on the spheres radius: They may range across studies
# from 4 mm to 12 mm... One way is to explore the homogenity
# of times-series within the sphere.

pcc_coords = dmn_coords[0]
##########################################################################
# We start by computing the grey matter voxel-wise time series.

import numpy as np
from nilearn.image.resampling import coord_transform
gm_masker = input_data.NiftiMasker(
    mask_img=icbm152_grey_mask, standardize=True, #low_pass=0.1, high_pass=0.01,
    t_r=2.5, detrend=True, memory='nilearn_cache', memory_level=1, verbose=2)
gm_voxels_time_series = gm_masker.fit_transform(func_filename,
                                                confounds=[confound_filename])

###############################################################################
# Then compute voxels indices within the grey mask
gm_mask_img = gm_masker.mask_img
gm_mask_array = gm_mask_img.get_data()
i, j, k = np.where(gm_mask_array != 0)
gm_mask_mni_coords = np.array(coord_transform(i, j, k, gm_mask_img.affine)).T

###############################################################################
# and use them to extract the ones near/laying into within the PCC sphere 8mm
sphere_8mm_mask = np.linalg.norm(gm_mask_mni_coords - pcc_coords, axis=1) <= 8.
pcc_8mm_voxels_time_series = gm_voxels_time_series[:, sphere_8mm_mask]

print(pcc_8mm_voxels_time_series.shape)
###############################################################################
# We can display the time-series of 8 random voxels and the average time series
plt.figure()
plt.plot(pcc_8mm_voxels_time_series[:, [3, 9, 16, 292, 531, 861, 150, 1708]])
plt.title('PCC voxel-wise time-series')
###############################################################################
# Within spheres seed-to-voxel correlations
# -----------------------------------------

###############################################################################
# We can compute the seed-to-voxel correlation within the sphere
pcc_8mm_mean_time_series = dmn_8mm_average_time_series[:, 0]
pcc_8mm_seed_to_voxel_correlation = np.dot(
    pcc_8mm_voxels_time_series.T,
    pcc_8mm_mean_time_series) / pcc_8mm_mean_time_series.shape[0]
array = np.zeros(sphere_8mm_mask.shape)
array[sphere_8mm_mask] = pcc_8mm_seed_to_voxel_correlation
pcc_8mm_seed_to_voxel_correlation_img = gm_masker.inverse_transform(
    array)

###############################################################################
# We can repeat the operation looping over the remaining seeds
seeds_8mm_to_voxel_correlation_imgs = []
for seed_coords, seed_8mm_mean_time_series in zip(
        dmn_coords[1:], dmn_8mm_average_time_series[:, 1:]):
    sphere_8mm_mask = np.linalg.norm(
        gm_mask_mni_coords - seed_coords, axis=1) <= 8.
    seed_8mm_voxels_time_series = gm_voxels_time_series[:, sphere_8mm_mask]
    seed_8mm_to_voxel_correlation = np.dot(
        seed_8mm_voxels_time_series.T,
        seed_8mm_mean_time_series) / seed_8mm_mean_time_series.shape[0]
    array = np.zeros(sphere_8mm_mask.shape)
    array[sphere_8mm_mask] = seed_8mm_to_voxel_correlation
    seeds_8mm_to_voxel_correlation_imgs.append(
        gm_masker.inverse_transform(array))


###############################################################################
# and plot the associated correlation map.
title = 'DMN 8mm seed-to-within seed voxels correlation maps'
display = plotting.plot_stat_map(pcc_8mm_seed_to_voxel_correlation_img,
                                 title=title, display_mode='z')

# Now add as an overlay the maps for the MPFC and the left and right TPJ
for seed_8mm_to_voxel_correlation_img in seeds_8mm_to_voxel_correlation_imgs:
    display.add_overlay(seed_8mm_to_voxel_correlation_img)

###############################################################################
# Let's now repeat the same operation with 5mm radius.
    sphere_mask = np.linalg.norm(gm_mask_mni_coords - pcc_coords, axis=1) <= radius
    pcc_voxels_time_series = gm_voxels_time_series[:, sphere_mask]
    masker = input_data.NiftiSpheresMasker(
        [pcc_coords], radius=radius, mask_img=icbm152_grey_mask,
        detrend=True, #low_pass=0.1, high_pass=0.01, t_r=2.5, 
        standardize=True,
        memory='nilearn_cache', memory_level=1, verbose=2)
    pcc_mean_time_series = masker.fit_transform(func_filename,
                                                confounds=[confound_filename])
    pcc_seed_to_voxel_correlation = np.dot(
        pcc_voxels_time_series.T,
        pcc_mean_time_series) / pcc_mean_time_series.shape[0]
    array = np.zeros(sphere_mask.shape)
    array[sphere_mask] = pcc_seed_to_voxel_correlation
    pcc_seed_to_voxel_correlation_img = gm_masker.inverse_transform(
        array)
    figure = plt.figure(figsize=(8, 8))
    title = 'PCC {0}mm, {1} voxels'.format(radius,
                                           pcc_voxels_time_series.shape[1])
    plotting.plot_stat_map(
        pcc_seed_to_voxel_correlation_img, cut_coords=[18],
        title=title, display_mode='z', figure=figure)

###############################################################################
# Radius of 5mm seems preferable.

###############################################################################
# Connectome new try with better radius
# -------------------------------------

masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=5., mask_img=icbm152_grey_mask,
    detrend=True, #low_pass=0.1, high_pass=0.01, t_r=2.5,
    standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)
dmn_spheres_time_series = masker.fit_transform(func_filename,
                                               confounds=[confound_filename])
estimator = LedoitWolf()
estimator.fit(dmn_spheres_time_series)
precision_matrix = estimator.precision_
plotting.plot_connectome(-precision_matrix, dmn_coords,
                         title="Default Mode Network Connectivity",
                         display_mode='lyrz', edge_kwargs={'lw': 4.})
plotting.show()

###############################################################################
# We succeed to recover better hemispheric connection, and observe
# anti-correlation between left and right regions of temporo-parietal junction.
