"""
Extracting resting-state signals from different atlases for comparison
======================================================================

In this example :class:`nilearn.input_data.NiftiLabelsMasker` is used to
extract time series from nifti objects using different parcellation atlases.

The time series of all subjects of the ADHD Dataset are concatenated to create
parcel-wise correlation matrices for each atlas.

"""

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
from matplotlib import pyplot as plt

# load atlases
destrieux = datasets.fetch_atlas_destrieux_2009()
yeo = datasets.fetch_atlas_yeo_2011()
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

print('Destrieux atlas nifti image (3D) is located at: %s' % destrieux['maps'])
print('Yeo atlas nifti image (3D) with 17 parcels and liberal mask is located '
      'at: %s' % yeo['thick_17'])
print('Harvard Oxford atlas nifti image (3D) thresholded at .25 is located '
      'at: %s' % harvard_oxford['maps'])

atlases = {'Destrieux Atlas (struct)': destrieux['maps'],
           'Yeo Atlas 17 thick (func)': yeo['thick_17'],
           'Harvard Oxford > 25% (struct)': harvard_oxford['maps']}

# load functional data
data = datasets.fetch_adhd(n_subjects=10)

print('Functional nifti images (4D, one per subject) are located at : %r'
      % data['func'])
print('Counfound csv files (one per subject) are located at : %r'
      % data['confounds'])

# loop over atlases
for name, atlas in sorted(atlases.items()):
    # create masker to extract functional data within atlas parcels
    masker = NiftiLabelsMasker(labels_img=atlas,
                               standardize=True,
                               memory='nilearn_cache')

    # extract time series from all subjects and concatenate them
    time_series = []
    for func, confounds in zip(data.func, data.confounds):
        time_series.append(masker.fit_transform(func, confounds=confounds))

    time_series = np.concatenate(time_series)

    # calculate correlation matrix and display
    correlation_matrix = np.corrcoef(time_series.T)

    plt.figure(figsize=(5, 5))
    plt.suptitle(name, size=14)
    plt.imshow(correlation_matrix, interpolation="nearest")

plt.show()
