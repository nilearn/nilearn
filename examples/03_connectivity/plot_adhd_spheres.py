"""
Extracting brain signal from spheres
====================================

This example extract brain signals from spheres described by the coordinates
of their center in MNI space and a given radius in millimeters. In particular,
this example extracts signals from Default Mode Network regions and compute a
connectome from them.

"""

##########################################################################
# Retrieve the dataset
# ---------------------
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


##########################################################################
# Coordinates of Default Mode Network
# ------------------------------------
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (1, 50, -5)]
labels = [
          'Posterior Cingulate Cortex',
          'Left Temporoparietal junction',
          'Right Temporoparietal junction',
          'Medial prefrontal cortex',
         ]


##########################################################################
# Extracts signal from sphere around DMN seeds
# ---------------------------------------------
from nilearn import input_data

masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

time_series = masker.fit_transform(func_filename,
                                   confounds=[confound_filename])

##########################################################################
# Display time series
# --------------------
import matplotlib.pyplot as plt
for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()


##########################################################################
# Compute partial correlation matrix
# -----------------------------------
# Using object :class:`nilearn.connectome.ConnectivityMeasure`: Its
# default covariance estimator is Ledoit-Wolf, allowing to obtain accurate
# partial correlations.
from nilearn.connectome import ConnectivityMeasure
connectivity_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrix = connectivity_measure.fit_transform(
    [time_series])[0]

##########################################################################
# Display connectome
# -------------------
from nilearn import plotting

plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Default Mode Network Connectivity")

##########################################################################
# Display connectome with hemispheric projections.
# Notice (0, -52, 18) is included in both hemispheres since x == 0.
plotting.plot_connectome(partial_correlation_matrix, dmn_coords,
                         title="Connectivity projected on hemispheres",
                         display_mode='lyrz')

plotting.show()
