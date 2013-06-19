"""
Simple NeuroImaging data visualization and manipulation
========================================================

Simple example to show data manipulation and visualization.
"""

# Fetch data ################################################################
from nilearn import datasets
haxby_files = datasets.fetch_haxby_simple()

# Get the file names relative to this dataset
bold = haxby_files.func

# Load the NIfTI data
import nibabel
nifti_img = nibabel.load(bold)
fmri_data = nifti_img.get_data()
fmri_affine = nifti_img.get_affine()

# Visualization #############################################################
import numpy as np
import pylab as pl

# Compute the mean EPI: we do the mean along the axis 3, which is time
mean_img = np.mean(fmri_data, axis=3)

# pl.figure() creates a new figure
pl.figure(figsize=(7, 4))

# First subplot: coronal view
# subplot: 1 line, 3 columns and use the first subplot
pl.subplot(1, 3, 1)
# Turn off the axes, we don't need it
pl.axis('off')
# We use pl.imshow to display an image, and use a 'gray' colormap
# we also use np.rot90 to rotate the image
pl.imshow(np.rot90(mean_img[:, 32, :]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.title('Coronal')

# Second subplot: sagittal view
pl.subplot(1, 3, 2)
pl.axis('off')
pl.title('Sagittal')
pl.imshow(np.rot90(mean_img[15, :, :]), interpolation='nearest',
          cmap=pl.cm.gray)

# Third subplot: axial view
pl.subplot(1, 3, 3)
pl.axis('off')
pl.title('Axial')
pl.imshow(np.rot90(mean_img[:, :, 32]), interpolation='nearest',
          cmap=pl.cm.gray)
pl.subplots_adjust(left=.02, bottom=.02, right=.98, top=.95,
                   hspace=.02, wspace=.02)

# Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nilearn.masking import compute_epi_mask
mask_img = compute_epi_mask(nifti_img)
mask_data = mask_img.get_data().astype(bool)

# We create a new figure
pl.figure(figsize=(3, 4))
# A plot the axial view of the mask to compare with the axial
# view of the raw data displayed previously
pl.axis('off')
pl.imshow(np.rot90(mask_data[:, :, 32]), interpolation='nearest')
pl.subplots_adjust(left=.02, bottom=.02, right=.98, top=.95)

# Applying the mask #########################################################

from nilearn.masking import apply_mask
masked_data = apply_mask(nifti_img, mask_img)

# masked_data shape is (instant number, voxel number). We can plot the first 10
# lines: they correspond to timeseries of 10 voxels on the side of the
# brain
pl.figure(figsize=(7, 5))
pl.plot(masked_data[:10].T)
pl.xlabel('Time', fontsize=16)
pl.ylabel('Voxel', fontsize=16)
pl.xlim(0, 22200)
pl.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)
pl.show()
