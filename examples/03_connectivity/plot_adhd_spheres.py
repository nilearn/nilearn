"""
Extracting brain signal from spheres
====================================

This example extracts brain signals from spheres described by the coordinates
of their center in MNI space and a given radius in millimeters. In particular,
this example extracts signals from Default Mode Network regions and computes
partial correlations between them.

"""

##########################################################################
# Data preparation
#-----------------

##########################################################################
# We retrieve the first subject data of the ADHD dataset.
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


##########################################################################
# We manually give coordinates of 4 regions from the Default Mode Network.
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
    'Posterior Cingulate Cortex',
    'Left Temporoparietal junction',
    'Right Temporoparietal junction',
    'Medial prefrontal cortex'
]

##########################################################################
# Signals extraction
#-------------------

##########################################################################
# It is advised to intersect the spheric regions with a grey matter mask.
# Since we do not have subject grey mask, we resort to a less precise
# group-level one.
icbm152_grey_mask = datasets.fetch_icbm152_brain_gm_mask()

##########################################################################
# We extracts signal from sphere around DMN seeds.
from nilearn import input_data

masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=6, mask_img=icbm152_grey_mask,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])

##########################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt
plt.figure()
for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Preprocessed signal')
plt.legend()
plt.tight_layout()


##########################################################################
# Partial correlations estimation
#--------------------------------

##########################################################################
# We can compute partial correlation matrix using object
# :class:`nilearn.connectome.ConnectivityMeasure`. We keep its default
# covariance estimator Ledoit-Wolf, since it computes accurate partial
# correlations when the number of ROIs is small.
from nilearn.connectome import ConnectivityMeasure
connectivity_measure = ConnectivityMeasure(kind='partial correlation')
print(connectivity_measure)


##########################################################################
# *connectivity_measure* accepts a list of 2D time-series arrays from several
# subjects and returns a list of their individual connectivity matrices.
# So here we provide it with a one element list containing our subject ROIs
# time-series, and then pick the computed matrix from the output list.
partial_correlation_matrix = connectivity_measure.fit_transform(
    [time_series])[0]

##########################################################################
# We can check that we got a square (n_spheres, n_spheres) connectivity matrix.
print('partial correrlation matrix has shape {0}'.format(
    partial_correlation_matrix.shape))

##########################################################################
# Visualizing the connections
#----------------------------

##########################################################################
# Display connectome
from nilearn import plotting

plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Default Mode Network Connectivity",
                         colorbar=True, edge_vmax=.5)

# Display connectome with hemispheric projections.
# Notice (0, -52, 18) is included in both hemispheres since x == 0.
title = "Connectivity projected on hemispheres"
plotting.plot_connectome(partial_correlation_matrix, dmn_coords, title=title,
                         display_mode='lyrz')

##########################################################################
# Hesitating on the radius? Check the homogeniety of your signals!
#-----------------------------------------------------------------

from nilearn.input_data.nifti_spheres_masker import _iter_signals_from_spheres
import nibabel
from nilearn import signal
from sklearn.covariance import EmpiricalCovariance
import numpy as np
#radii = np.arange(15) + 1.
#radii = [2., 6., 8., 12., 20., 100.]
radii = [2., 6., 12.]
seed_based_correlation_img = {}
gm_masker = input_data.NiftiMasker(
    mask_img=icbm152_grey_mask,
    detrend=True, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)
gm_voxels_time_series = gm_masker.fit_transform(
    func_filename, confounds=[confound_filename])
for radius in radii:
    seed_masker = input_data.NiftiSpheresMasker(
        dmn_coords[:1], radius=radius, mask_img=icbm152_grey_mask,
        detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5, standardize=True,
        memory='nilearn_cache', memory_level=1, verbose=2)
    seed_signal = seed_masker.fit_transform(
        func_filename, confounds=[confound_filename])
    seed_based_correlations = np.dot(
        gm_voxels_time_series.T, seed_signal) / seed_signal.shape[0]
    seed_based_correlation_img[radius] = gm_masker.inverse_transform(
        seed_based_correlations.T)

# Form seed image
raw_gm_masker = input_data.NiftiMasker(
    mask_img=icbm152_grey_mask, standardize=True,
    memory='nilearn_cache', memory_level=1, verbose=2)
raw_gm_voxels_time_series = raw_gm_masker.fit_transform(func_filename)
sphere_indicator_img = {}
for radius in radii:
    raw_seed_masker = input_data.NiftiSpheresMasker(
        dmn_coords[:1], standardize=True, radius=radius,
        mask_img=icbm152_grey_mask,
        memory='nilearn_cache', memory_level=1, verbose=1)
    raw_seed_masker.fit()
    raw_sphere_time_series = _iter_signals_from_spheres(
        raw_seed_masker.seeds_, nibabel.load(func_filename),
        raw_seed_masker.radius,
        raw_seed_masker.allow_overlap,
        mask_img=raw_seed_masker.mask_img).next()
    raw_sphere_to_gm_correlations = np.dot(
        raw_gm_voxels_time_series.T,
        (raw_sphere_time_series / raw_sphere_time_series.std(axis=0))) /\
        raw_sphere_time_series.shape[0]
    raw_sphere_gm_max_correlations = raw_sphere_to_gm_correlations.max(axis=1)
    max_corr = raw_sphere_gm_max_correlations.max()
    sphere_indicator = np.zeros(raw_sphere_gm_max_correlations.shape)
    indices = np.argsort(raw_sphere_gm_max_correlations)[::-1]
    for k in np.arange(10)[::-1] + 1:
        sphere_indicator[indices[:k * raw_sphere_time_series.shape[1]]] = k

    sphere_indicator = np.zeros(raw_sphere_gm_max_correlations.shape)
    for n, ts in enumerate(raw_sphere_time_series.T):
        print n
        raw_ts_to_gm_correlations = np.dot(
            raw_gm_voxels_time_series.T, ts / ts.std()) / ts.shape[0]
        index = np.argmax(raw_ts_to_gm_correlations)
        sphere_indicator[index] = 1.

#    assert(np.sum(sphere_indicator) == np.shape(raw_sphere_time_series)[1])
    sphere_indicator_img[radius] = raw_gm_masker.inverse_transform(
        sphere_indicator)

from nilearn import image
mean_func_img = image.mean_img(func_filename)

for radius in [6.]:
    display = plotting.plot_stat_map(seed_based_correlation_img[radius],
                                     bg_img=mean_func_img,
                                     cut_coords=dmn_coords[0], threshold=.25,
                                     title=radius)
    display.add_contours(sphere_indicator_img[radius])

for radius in radii:
    plotting.plot_roi(sphere_indicator_img[radius], cut_coords=dmn_coords[0],
                      title=radius)

plotting.show()
stop
# Compute seed-to-voxel correlations within the spheres

from nilearn.input_data.nifti_spheres_masker import _iter_signals_from_spheres
import nibabel
from nilearn import signal
from sklearn.covariance import EmpiricalCovariance
import numpy as np
seed_time_series = {}
cleaned_spheres = {}
seed_masker = {}
#radii = np.arange(15) + 1.
radii = [2., 6., 8., 12., 20., 100.]
seed_based_correlations = {}
vox_to_vox_covariance = {}
for radius in radii:
    seed_masker[radius] = input_data.NiftiSpheresMasker(
        dmn_coords[:1], radius=radius,
        mask_img=icbm152_grey_mask,
        detrend=True, standardize=False,
        memory='nilearn_cache', memory_level=1, verbose=0)
    seed_masker[radius].fit()
    sphere = _iter_signals_from_spheres(
        seed_masker[radius].seeds_, nibabel.load(func_filename),
        seed_masker[radius].radius,
        seed_masker[radius].allow_overlap,
        mask_img=seed_masker[radius].mask_img).next()
    cleaned_spheres[radius] = signal.clean(
        sphere, confounds=[confound_filename], standardize=False)
#    cleaned_spheres[radius] = sphere
    seed_time_series[radius] = seed_masker[radius].fit_transform(
        func_filename, confounds=[confound_filename])
#    seed_based_correlations[radius] = np.dot(cleaned_spheres[radius].T,
#                                     seed_time_series[radius]) / \
#        seed_time_series[radius].shape[0]
    time_series = cleaned_spheres[radius]
    time_series -= time_series.mean(axis=1)[:, np.newaxis]
    ts_ranks = np.sum(time_series ** 2, axis=0)
    n_times, n_voxels = time_series.shape
    constant = (n_voxels ** 2) * n_times * (n_times ** 2 - 1) / (12. * (
        n_voxels - 1))
    seed_based_correlations[radius] = 1. - ts_ranks / constant
    seed_based_correlations[radius] *= 100.

    covariance_estimator = EmpiricalCovariance()
    covariance_estimator.fit(cleaned_spheres[radius])
    vox_to_vox_covariance[radius] = covariance_estimator.covariance_


plt.figure()
plt.boxplot([np.triu(vox_to_vox_covariance[radius], k=1) for radius in radii],
            whis=np.inf)
plt.xticks(np.arange(len(radii)) + 1, radii)
plt.show()
stop
# Create a mask describing one masked spheric ROI

##########################################################################
# There is no concensus on the spheres radius. They may range from 4 mm to
# 12 mm... One way is to use Kendall coefficient [1] to explore the homogenity
# of times-series within the sphere.

masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=12., mask_img=icbm152_grey_mask,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])

gm_masker = input_data.NiftiMasker(
    mask_img=icbm152_grey_mask,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

gm_time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])
print time_series.shape
time_series -= time_series.mean(axis=1)[:, np.newaxis]

#    from sklearn import covariance
#    covariance_estimator = covariance.EmpiricalCovariance()
#    covariance_estimator.fit(time_series.T)
#    covariance_matrix = covariance_estimator.covariance_
#    ts_ranks = covariance_matrix.sum(axis=0)
ts_ranks = np.sum(time_series ** 2, axis=0)
n_times, n_voxels = time_series.shape
constant = (n_voxels ** 2) * n_times * (n_times ** 2 - 1) / (12. * (
    n_voxels - 1))
kundall[radius] = 1. - ts_ranks / constant

partial_correlation_matrix2 = connectivity_measure.fit_transform(
    [time_series])[0]

plotting.plot_connectome(partial_correlation_matrix2, dmn_coords,
                         title="12 mm radius", edge_vmax=.5, colorbar=True)

plotting.show()
