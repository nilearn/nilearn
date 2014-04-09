"""
NeuroImaging volumes visualization
====================================

Simple example to show Nifti data visualization.
"""

### Fetch data ################################################################

from nilearn import datasets

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
mean_img = np.mean(fmri_data, axis=3)
# Note that this can also be done on Nifti images using
# nilearn.image.mean_img

# plt.figure() creates a new figure
plt.figure(figsize=(7, 4))

# First subplot: coronal view
# subplot: 1 line, 3 columns and use the first subplot
plt.subplot(1, 3, 1)
# Turn off the axes, we don't need it
plt.axis('off')
# We use plt.imshow to display an image, and use a 'gray' colormap
# we also use np.rot90 to rotate the image
plt.imshow(np.rot90(mean_img[:, 32, :]), interpolation='nearest',
          cmap=plt.cm.gray)
plt.title('Coronal')

# Second subplot: sagittal view
plt.subplot(1, 3, 2)
plt.axis('off')
plt.title('Sagittal')
plt.imshow(np.rot90(mean_img[15, :, :]), interpolation='nearest',
          cmap=plt.cm.gray)

# Third subplot: axial view
plt.subplot(1, 3, 3)
plt.axis('off')
plt.title('Axial')
plt.imshow(np.rot90(mean_img[:, :, 32]), interpolation='nearest',
          cmap=plt.cm.gray)
plt.subplots_adjust(left=.02, bottom=.02, right=.98, top=.95,
                   hspace=.02, wspace=.02)

### Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nilearn.masking import compute_epi_mask
mask_img = compute_epi_mask(haxby_files.func[0])
mask_data = mask_img.get_data().astype(bool)

# We create a new figure
plt.figure(figsize=(3, 4))
# A plot the axial view of the mask to compare with the axial
# view of the raw data displayed previously
plt.axis('off')
plt.imshow(np.rot90(mask_data[:, :, 32]), interpolation='nearest')
plt.subplots_adjust(left=.02, bottom=.02, right=.98, top=.95)

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

