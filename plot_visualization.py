"""
Simple NeuroImaging data visualization and manipulation
========================================================

Simple example to show data manipulation and visualization.
"""

# Fetch data ################################################################
from nisl import datasets
haxby = datasets.fetch_haxby()

# Get the files relative to this dataset
files = haxby.files
bold = files[1]

# Load the NIfTI data
import nibabel
nifti_img = nibabel.load(bold)
data = nifti_img.get_data()

# Visualization #############################################################
import numpy as np
import pylab as pl
pl.figure()
pl.subplot(131)
pl.axis('off')
pl.title('Coronal')
pl.imshow(np.rot90(data[:, 32, :, 100]), interpolation='nearest',
          cmap=pl.cm.gray)

pl.subplot(132)
pl.axis('off')
pl.title('Sagittal')
pl.imshow(np.rot90(data[15, :, :, 100]), interpolation='nearest',
          cmap=pl.cm.gray)

pl.subplot(133)
pl.axis('off')
pl.title('Axial')
pl.imshow(np.rot90(data[:, :, 32, 100]), interpolation='nearest',
          cmap=pl.cm.gray)

# Extracting a brain mask ###################################################

# Simple computation of a mask from the fMRI data
from nisl.masking import compute_mask
mask = compute_mask(data)

# We create a new figure
pl.figure()
# A plot the axial view of the mask to compare with the axial
# view of the raw data displayed previously
pl.imshow(np.rot90(mask[:, :, 32]), interpolation='nearest')

# Applying the mask #########################################################

# Applying the mask is just a simple array manipulation
masked_data = data[mask]

# masked_data is now a voxel x time matrix. We can plot the first 10
# lines: they correspond to time-series of 10 voxels on the side of the
# brain
pl.figure()
pl.plot(masked_data[:10].T)
pl.xlabel('Time')
pl.ylabel('Voxel')

pl.show()
