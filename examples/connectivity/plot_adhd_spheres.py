"""
Extracting brain signal from spheres
====================================

This example extract brain signals from spheres described by the coordinates
of their center in MNI space and a given radius in millimeters. In particular,
this example extracts signals from Default Mode Network regions and compute a
connectome from them.

"""

# Fetching dataset ############################################################
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


# Extracting region signals ###################################################
from nilearn import input_data


# Coordinates of Default Mode Network
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (0, 50, -5)]
labels = [
    'Posterior Cingulate Cortex',
    'Left Temporoparietal junction',
    'Right Temporoparietal junction',
    'Medial prefrontal cortex'
]

# Extracting signal from sphere around DMN seeds
masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=0.1, high_pass=0.01, t_r=2.5,
    memory='nilearn_cache', memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

time_series = masker.fit_transform(func_filename,
                             confounds=[confound_filename])


# Computing precision matrices ################################################
from sklearn.covariance import LedoitWolf
cve = LedoitWolf()
cve.fit(time_series)

# Displaying results ##########################################################
import matplotlib.pyplot as plt
from nilearn import plotting

# Display time series
for time_serie, label in zip(time_series.T, labels):
    plt.plot(time_serie, label=label)

plt.title('Default Mode Network Time Series')
plt.xlabel('Scan number')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()

# Display connectome
title = "Default Mode Network Connectivity"
plotting.plot_connectome(cve.precision_, dmn_coords, title=title)
plt.show()
