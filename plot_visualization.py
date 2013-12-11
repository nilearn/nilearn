"""
Simple NeuroImaging data visualization and manipulation
========================================================

Simple example to show Nifti data manipulation and visualization.
"""

### Coordinates of the selected slice #########################################

c = 27
s = 21
a = 26

### Fetch data ################################################################

from nilearn import datasets

haxby_files = datasets.fetch_haxby(n_subjects=1)

# Take a look at the files proposed for this dataset
print 'Haxby files:', haxby_files.keys()
# Haxby files: ['mask_house_little', 'anat', 'mask_house', 'mask_face', 'func',
#               'session_target', 'mask_vt', 'mask_face_little']

print haxby_files.func[0]
# /path/to/nisl_data/haxby2001_simple/pymvpa-exampledata/bold.nii.gz

### Load an fMRI file #########################################################

import nibabel

fmri_img = nibabel.load(haxby_files.func[0])
fmri_data = fmri_img.get_data()
fmri_affine = fmri_img.get_affine()

### Load a text file ##########################################################

import numpy

labels = numpy.genfromtxt(haxby_files.session_target[0], skip_header=1,
                          usecols=[0], dtype=basestring)
print numpy.unique(labels)
# array(['bottle', 'cat', 'chair', 'face', 'house', 'rest', 'scissors',
#        'scrambledpix', 'shoe'], dtype=object)

### Visualization #############################################################

import numpy as np
import matplotlib.pyplot as plt

# Compute the mean EPI: we do the mean along the axis 3, which is time
mean_img = np.mean(fmri_data, axis=3)

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

### Visualization function ####################################################

import matplotlib.pyplot as plt


# One-liner to display brain slices
def plot_brain(brain, x, y, z, title, cmap='hot'):
    plt.figure(figsize=(7, 4))
    plt.subplot(1, 3, 1)
    plt.axis('off')
    brain = brain.copy()
    plt.imshow(brain[x, :, :].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Coronal')
    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.imshow(brain[:, y, :].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Sagittal')
    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.imshow(brain[:, :, z].T, origin='lower', interpolation='nearest',
               cmap=cmap)
    plt.title('Axial')
    plt.suptitle(title)
    plt.tight_layout()


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

### Find voxels of interest ###################################################

# Smooth the data
from nilearn.image import smooth_img
fmri_img = smooth_img(haxby_files.func[0], fwhm=6)
fmri_data = fmri_img.get_data()
plot_brain(fmri_data.mean(axis=-1), c, s, a, 'Smoothed mean EPI')

# Run a T-test for face and houses
from scipy import stats
_, pvalues = stats.ttest_ind(fmri_data[..., labels == 'face'],
                             fmri_data[..., labels == 'house'], axis=-1)

# Use a log scale for p-values
pvalues = - numpy.log10(pvalues)
pvalues[numpy.isnan(pvalues)] = 0.
pvalues[pvalues > 10] = 10
plot_brain(pvalues, c, s, a, 'p-values')

### Build a mask ##############################################################

# Thresholding
pvalues[pvalues < 5] = 0
plot_brain(pvalues, c, s, a, 'Thresholded p-values')

# Binarization and intersection with VT mask
pvalues = (pvalues != 0)
vt = nibabel.load(haxby_files.mask_vt[0]).get_data().astype(bool)
pvalues = numpy.logical_and(pvalues, vt)
plot_brain(pvalues, c, s, a, 'Intersection with ventral temporal mask')

# Dilation
from scipy.ndimage import binary_dilation
pvalues = binary_dilation(pvalues)
plot_brain(pvalues, c, s, a, 'Dilated mask')

# Identification of connected components
from scipy.ndimage import label
labels, n_labels = label(pvalues)
plot_brain(labels, c, s, a, 'Connected components', 'gnuplot')

# Save the result
nibabel.save(nibabel.Nifti1Image(labels, fmri_affine), 'mask.nii')
plt.show()
