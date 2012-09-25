"""
Example of automatic mask computation
"""

import pylab as pl
import numpy as np
from nibabel import Nifti1Image

from nisl import datasets, io, utils

# Load Haxby dataset
haxby = datasets.fetch_haxby()
haxby_img = utils.check_niimg(haxby.func)
# Restrict haxby to 150 frames to speed up computation
haxby_func = np.rot90(haxby_img.get_data()[..., :150])
haxby_img = Nifti1Image(haxby_func, haxby_img.get_affine()) 
# Load mask provided by Haxby
haxby_mask = np.rot90(utils.check_niimg(haxby.mask).get_data().astype(np.bool))

# Display helper
background = np.mean(haxby_func, axis=-1)[..., 27]
def display_mask(background, mask, title):
    pl.axis('off')
    pl.imshow(background, interpolation='nearest', cmap=pl.cm.gray)
    ma = np.ma.masked_equal(mask, False)
    pl.imshow(ma, interpolation='nearest', cmap=pl.cm.autumn, alpha=0.5)
    pl.title(title)

# Generate mask with default parameters
masker = io.NiftiMasker()
masker.fit(haxby_img)
default_mask = masker.mask_.get_data().astype(np.bool)
pl.figure()
display_mask(background, default_mask[..., 27], 'Default mask')
pl.show()

# Generate mask with opening
masker = io.NiftiMasker(mask_opening=True)
masker.fit(haxby_img)
opening_mask = masker.mask_.get_data().astype(np.bool)
pl.figure()
display_mask(background, opening_mask[..., 27], 'Mask with opening')
pl.show()

# Generate mask with upper cutoff
masker = io.NiftiMasker(mask_opening=True, mask_upper_cutoff=0.8)
masker.fit(haxby_img)
cutoff_mask = masker.mask_.get_data().astype(np.bool)

# Plot the mask and compare it to original
pl.figure()
pl.subplot(1, 2, 1)
display_mask(background, haxby_mask[..., 27], 'Haxby mask')

pl.subplot(1, 2, 2)
display_mask(background, cutoff_mask[..., 27], 'Mask with cutoff')
pl.subplots_adjust(top=0.8)
pl.show()
