"""
Extracting brain signal from spheres
====================================

This example estimates a connectivity between Default Mode Network components
using spheres as ROIs.

"""


# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


# Extracting region signals ###################################################
from nilearn import image
from nilearn import input_data

from sklearn.externals.joblib import Memory
mem = Memory('nilearn_cache')

print("... Extracting time series ...")

# Coordinates of Default Mode Network
dmn_coords = [(0, -52, 18), (-46, -68, 32), (46, -68, 32), (0, 50, -5)]
labels = [
    'Posterior Cingulate Cortex',
    'Left Temporoparietal junction',
    'Right Temporoparietal junction',
    'Medial prefrontal cortex'
]

masker = input_data.NiftiSpheresMasker(
    dmn_coords, radius=8,
    detrend=True, standardize=True,
    low_pass=None, high_pass=0.01, t_r=2.5,
    memory=mem, memory_level=1, verbose=2)

func_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

# Computing some confounds
hv_confounds = mem.cache(image.high_variance_confounds)(func_filename)

time_series = masker.fit_transform(func_filename,
                             confounds=[hv_confounds, confound_filename])


# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from sklearn.covariance import LedoitWolf
cve = LedoitWolf()
cve.fit(time_series)

# Displaying results ##########################################################
import matplotlib.pyplot as plt
from nilearn import plotting

# Define colors to harmonize them among plots
colors = ['b', 'g', 'r', 'm']

# Display time series
for time_serie, label, color in zip(time_series.T, labels, colors):
    plt.plot(time_serie, label=label, color=color)

plt.title('Default Mode Network Time Series')
plt.xlabel('Time')
plt.ylabel('Normalized signal')
plt.legend()
plt.tight_layout()

# Display connectome
title = "Default Mode Network Connectivity"
plotting.plot_connectome(cve.precision_, dmn_coords, title=title,
                         node_color=colors)
plt.show()
