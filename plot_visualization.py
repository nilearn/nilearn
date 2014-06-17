"""
NeuroImaging volumes visualization
====================================

Simple example to show Nifti data visualization.
"""

### Fetch data ################################################################

from nilearn import datasets
from nilearn.image.image import mean_img
from nilearn.plotting.img_plotting import plot_epi, plot_roi
import matplotlib as mpl

haxby_files = datasets.fetch_haxby(n_subjects=1)

### Load an fMRI file #########################################################

import nibabel

fmri_img = nibabel.load(haxby_files.func[0])
fmri_data = fmri_img.get_data()
fmri_affine = fmri_img.get_affine()

### Visualization #############################################################

import numpy as np
import matplotlib.pyplot as plt

# Compute the mean EPI: we do the mean along the axis 3, which is time
mean_haxby = mean_img(haxby_files.func)

plot_epi(mean_haxby)

### Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nilearn.masking import compute_epi_mask
mask_img = compute_epi_mask(haxby_files.func[0])
mask_data = mask_img.get_data().astype(bool)

plot_roi(mean_haxby, mask_img)

### Applying the mask #########################################################

from nilearn.masking import apply_mask
masked_data = apply_mask(haxby_files.func[0], mask_img)

# masked_data shape is (instant number, voxel number). We can plot the first 10
# lines: they correspond to timeseries of 10 voxels on the side of the
# brain
plt.figure(figsize=(7, 5))
plt.plot(masked_data[:10].T)
plt.xlabel('Time', fontsize=16)
plt.ylabel('Voxel', fontsize=16)
plt.xlim(0, 22200)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)

plt.show()

