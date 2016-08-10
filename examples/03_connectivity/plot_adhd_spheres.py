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
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])

##########################################################################
# We display the time series and check visually their synchronization.
import matplotlib.pyplot as plt
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

##########################################################################
# There is no concensus on the spheres radius. They may range from 4 mm to
# 12 mm... One way is to use Kendall coefficient [1] to explore the homogenity
# of times-series within the sphere.

# We increase the radius to explore its impact on connectivity.
masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=12., mask_img=icbm152_grey_mask,
    detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])

partial_correlation_matrix2 = connectivity_measure.fit_transform(
    [time_series])[0]

plotting.plot_connectome(partial_correlation_matrix2, dmn_coords,
                         title="12 mm radius", edge_vmax=.5, colorbar=True)

plotting.show()
