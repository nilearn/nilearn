"""
Extracting signals from different atlases and compare correlation matrices
==========================================================================

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
destrieux_atlas = destrieux['maps']

yeo = datasets.fetch_atlas_yeo_2011()
yeo_atlas = yeo['thick_17']

harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
harvard_oxford_atlas = harvard_oxford['maps']

atlases = {'Destrieux Atlas': destrieux_atlas,
           'Yeo Atlas 17 thick': yeo_atlas,
           'Harvard Oxford > 25%': harvard_oxford_atlas}

# get functional data
data = datasets.fetch_adhd()

# loop over atlases
for name, atlas in sorted(atlases.items()):
    # create masker to extract functional data within atlas parcels
    masker = NiftiLabelsMasker(labels_img=atlas,
                               standardize=True,
                               memory='nilearn_cache')

    # load timeseries from all subjects and concatenate them
    time_series = []
    for func, confounds in zip(data.func, data.confounds):
        time_series.append(masker.fit_transform(func, confounds=confounds))

    time_series = np.concatenate((time_series))

    # calculate correlation matrix and append it to the list
    correlation_matrix = np.corrcoef(time_series.T)

    # display correlation matrices
    plt.figure(figsize=(5, 5))
    plt.suptitle(name, size=14)
    plt.imshow(correlation_matrix, interpolation="nearest")

plt.show()
