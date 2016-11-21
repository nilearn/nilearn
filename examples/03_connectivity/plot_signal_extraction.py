"""
Extracting signals from a brain parcellation
============================================

Here we show how to extract signals from a brain parcellation and compute
a correlation matrix.

We also show the importance of defining good confounds signals: the
first correlation matrix is computed after regressing out simple
confounds signals: movement regressors, white matter and CSF signals, ...
The second one is without any confounds: all regions are connected to
each other.

We explore the impact of different counfounds choices on voxel-to-voxel
connectivity and use it to evaluate the relevance of the chosen counfounds.

One reference that discusses the importance of confounds is `Varoquaux and
Craddock, Learning and comparing functional connectomes across subjects,
NeuroImage 2013
<http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.

This is just a code example, see the :ref:`corresponding section in the
documentation <parcellation_time_series>` for more.
"""

##############################################################################
# From images to an ROI-to-ROI correlation matrix
# -----------------------------------------------
#
# Retrieve the atlas and the functional data.
from nilearn import datasets

dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename, labels = dataset.maps, dataset.labels

print('Atlas ROIs are located in nifti image (3D) at: %s' %
      atlas_filename)  # 3D data

# One subject of resting-state data.
data = datasets.fetch_adhd(n_subjects=1)
fmri_filename = data.func[0]
confounds_filename = data.confounds[0]

##############################################################################
# Extract signals on a parcellation defined by labels using the
# NiftiLabelsMasker.
from nilearn.input_data import NiftiLabelsMasker
atlas_masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                                 memory='nilearn_cache', verbose=1)

# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = atlas_masker.fit_transform(fmri_filename,
                                         confounds=confounds_filename)

##############################################################################
# Compute and display a correlation matrix.
import numpy as np
correlation_matrix = np.corrcoef(time_series.T)

# Plot the correlation matrix
import numpy as np
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

# Add labels and adjust margins
x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)


##############################################################################
# Counfounding signals: a source of spurious correlations
# -------------------------------------------------------
#
# Same thing without confounds, to stress the importance of confounds.

time_series = atlas_masker.fit_transform(fmri_filename)
# Note how we did not specify confounds above. This is bad!

correlation_matrix = np.corrcoef(time_series.T)

# Mask the main diagonal for visualization:
np.fill_diagonal(correlation_matrix, 0)

plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, interpolation="nearest", cmap="RdBu_r",
           vmax=0.8, vmin=-0.8)

x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
plt.suptitle('No confounds', size=27)

##############################################################################
# Do my counfounds model noise properly? Voxel-to-voxel connectivity tells!
# -------------------------------------------------------------------------
#
# Check the relevance of chosen confounds: The distribution of voxel-to-voxel
# correlations should be tight and approximately centered to zero.

##############################################################################
# Firstly take a look at the confounds defined in the csv file.
from nilearn._utils.numpy_conversions import csv_to_array
confounds = csv_to_array(confounds_filename, names=True)
confounds_names = confounds.dtype.names
print('Csv file includes {0} confounds:\n{1}.'.format(
    len(confounds_names), ', '.join(confounds_names)))

##############################################################################
# Now compute voxel-wise time series with and without confounds removal,
# using NiftiMasker.
from nilearn.input_data import NiftiMasker
brain_masker = NiftiMasker(memory='nilearn_cache', verbose=1)
voxel_ts_raw = brain_masker.fit_transform(fmri_filename)
voxel_ts_cleaned_csv = brain_masker.fit_transform(fmri_filename,
                                                  confounds=confounds_filename)

##############################################################################
# For comparison, compute voxels signals after high variance confounds removal
from nilearn.image import high_variance_confounds
# We compute as many confounds as in the csv file
hv_confounds = high_variance_confounds(fmri_filename, n_confounds=17)
voxel_ts_cleaned_hv = brain_masker.fit_transform(fmri_filename,
                                                 confounds=hv_confounds)

##############################################################################
# Next, compute the voxel-to-voxel correlations

# Use only 1% of voxels, to save computation time
selected_voxels = range(0, voxel_ts_raw.shape[1], 100)
correlation_matrix_raw = np.corrcoef(voxel_ts_raw[:, selected_voxels].T)
correlation_matrix_cleaned_csv = np.corrcoef(
    voxel_ts_cleaned_csv[:, selected_voxels].T)
correlation_matrix_cleaned_hv = np.corrcoef(
    voxel_ts_cleaned_hv[:, selected_voxels].T)

##############################################################################
# and plot their histograms.
matrices = [correlation_matrix_raw, correlation_matrix_cleaned_csv,
            correlation_matrix_cleaned_hv]
labels = ['none', 'file confounds', 'high variance\nconfounds']
plt.figure(figsize=(8, 3))
for matrix, label, color in zip(matrices, labels, 'rgb'):
    plt.hist(matrix[np.triu_indices_from(matrix, k=1)],
             color=color, alpha=.4, bins=100, lw=0, label=label)

[ymin, ymax] = plt.ylim()
plt.vlines(0, ymin, ymax)
plt.legend(title='removed confounds')
plt.xlabel('voxel-to-voxel correlation values')
plt.tight_layout()

##############################################################################
# The morale: High variance confounds do not succeed to capture all noise
# sources. Motion regressors are mandatory !

plt.show()
