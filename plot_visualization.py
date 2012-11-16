"""
Simple NeuroImaging data visualization and manipulation
========================================================

Simple example to show data manipulation and visualization.
"""

# Fetch data ################################################################
from nisl import datasets
haxby_files = datasets.fetch_haxby_simple()

# Get the file names relative to this dataset
bold = haxby_files.func

# Load the NIfTI data
import nibabel
nifti_img = nibabel.load(bold)
fmri_data = nifti_img.get_data()

# Visualization #############################################################
import numpy as np
import pylab as pl

# Compute the mean EPI: we do the mean along the axis 3, which is time
mean_img = np.mean(fmri_data, axis=3)

# pl.figure() creates a new figure
pl.figure()

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

# Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nisl.masking import compute_epi_mask
mask = compute_epi_mask(mean_img)

# We create a new figure
pl.figure()
# A plot the axial view of the mask to compare with the axial
# view of the raw data displayed previously
pl.imshow(np.rot90(mask[:, :, 32]), interpolation='nearest')

# Applying the mask #########################################################

# Applying the mask is just a simple array manipulation
masked_data = fmri_data[mask]

# masked_data is now a voxel x time matrix. We can plot the first 10
# lines: they correspond to time-series of 10 voxels on the side of the
# brain
pl.figure()
pl.plot(masked_data[:10].T)
pl.xlabel('Time')
pl.ylabel('Voxel')

pl.show()
