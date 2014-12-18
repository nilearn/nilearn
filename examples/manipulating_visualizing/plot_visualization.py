"""
NeuroImaging volumes visualization
==================================

Simple example to show Nifti data visualization.
"""

### Fetch data ################################################################

from nilearn import datasets
from nilearn.image.image import mean_img
from nilearn.plotting.img_plotting import plot_epi, plot_roi

haxby_dataset = datasets.fetch_haxby(n_subjects=1)

### Visualization #############################################################

import matplotlib.pyplot as plt

# Compute the mean EPI: we do the mean along the axis 3, which is time
func_filename = haxby_dataset.func[0]
mean_haxby = mean_img(func_filename)

plot_epi(mean_haxby)

### Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nilearn.masking import compute_epi_mask
mask_img = compute_epi_mask(func_filename)

plot_roi(mask_img, mean_haxby)

### Applying the mask #########################################################

from nilearn.masking import apply_mask
masked_data = apply_mask(func_filename, mask_img)

# masked_data shape is (timepoints, voxels). We can plot the first 150
# timepoints from two voxels

plt.figure(figsize=(7, 5))
plt.plot(masked_data[:2, :150].T)
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.xlim(0, 150)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)

plt.show()
