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


One reference that discusses the importance of confounds is `Varoquaux and
Craddock, Learning and comparing functional connectomes across subjects,
NeuroImage 2013
<http://www.sciencedirect.com/science/article/pii/S1053811913003340>`_.

This is just a code example, see the :ref:`corresponding section in the
documentation <parcellation_time_series>` for more.
"""

from nilearn import datasets

# Retrieve our atlas
atlas_filename, labels = datasets.fetch_harvard_oxford('cort-maxprob-thr25-2mm')
print('Atlas ROIs are located in nifti image (4D) at: %s' %
      atlas_filename)  # 4D data

# And one subject of resting-state data
data = datasets.fetch_adhd(n_subjects=1)

# To extract signals on a parcellation defined by labels, we use the
# NiftiLabelsMasker
from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)


# Here we go from nifti files to the signal time series in a numpy
# array. Note how we give confounds to be regressed out during signal
# extraction
time_series = masker.fit_transform(data.func[0], confounds=data.confounds)

import numpy as np
correlation_matrix = np.corrcoef(time_series.T)

# Plot the correlation matrix
from matplotlib import pyplot as plt
plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, interpolation="nearest")

# Add labels and adjust margins
x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)


###############################################################################
# Same thing without confounds, to stress the importance of confounds

time_series = masker.fit_transform(data.func[0])

correlation_matrix = np.corrcoef(time_series.T)

plt.figure(figsize=(10, 10))
plt.imshow(correlation_matrix, interpolation="nearest")

x_ticks = plt.xticks(range(len(labels) - 1), labels[1:], rotation=90)
y_ticks = plt.yticks(range(len(labels) - 1), labels[1:])
plt.gca().yaxis.tick_right()
plt.subplots_adjust(left=.01, bottom=.3, top=.99, right=.62)
plt.suptitle('No confounds', size=27)

plt.show()
