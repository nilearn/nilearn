"""
Basic nilearn example
=====================

A simple example showing how to load an existing Nifti file and use
basic nilearn functionalities.
"""

# Import the os module, for file manipulation
import os

#########################################################################
# Let us use a Nifti file that is shipped with nilearn
from nilearn.datasets import data
anat_filename = os.path.join(os.path.dirname(data.__file__),
                             'avg152T1_brain.nii.gz')
print('anat_filename: %s' % anat_filename)

#########################################################################
# Using simple image nilearn functions
from nilearn import image
# functions containing 'img' can take either a filename or an image as input
smooth_anat_img = image.smooth_img(anat_filename, 3)

# While we are giving a file name as input, the object that is returned
# is a 'nibabel' object. It has data, and an affine
anat_data = smooth_anat_img.get_data()
print('anat_data has shape: %s' % str(anat_data.shape))
anat_affine = smooth_anat_img.get_affine()
print('anat_affineaffine:\n%s' % anat_affine)

# Finally, it can be passed to nilearn function
smooth_anat_img = image.smooth_img(smooth_anat_img, 3)

#########################################################################
# Visualization
from nilearn import plotting
cut_coords = (0, 0, 0)

# Like all functions in nilearn, plotting can be given filenames
plotting.plot_anat(anat_filename, cut_coords=cut_coords,
                   title='Anatomy image')

# Or nibabel objects
plotting.plot_anat(smooth_anat_img,
                   cut_coords=cut_coords,
                   title='Smoothed anatomy image')

#########################################################################
# Saving image to file
smooth_anat_img.to_filename('smooth_anat_img.nii.gz')

#########################################################################
# Finally, showing plots when used inside a terminal
plotting.show()
