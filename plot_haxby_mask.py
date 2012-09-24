"""
Example of automatic mask computation
"""

import pylab as pl
import numpy as np
from nisl import datasets, io, utils

# Load Haxby dataset
haxby = datasets.fetch_haxby()
haxby_func = utils.check_niimg(haxby.func).get_data()

# Load mask provided by Haxby
haxby_mask = utils.check_niimg(haxby.mask).get_data().astype(np.bool)
print "Haxby mask loss: %i" % np.where(haxby_func[-haxby_mask] != 0)[0].size

# Generate mask with default parameters
masker = io.NiftiMasker()
masker.fit(haxby.func)
default_mask = masker.mask_.get_data().astype(np.bool)
print "Default mask loss: %i" % np.where(haxby_func[-default_mask] != 0)[0].size

# Generate mask with opening
masker = io.NiftiMasker(mask_opening=True)
masker.fit(haxby.func)
opening_mask = masker.mask_.get_data().astype(np.bool)
print "Opening mask loss: %i" % np.where(haxby_func[-opening_mask] != 0)[0].size


# Generate mask with upper cutoff
masker = io.NiftiMasker(mask_opening=True, mask_upper_cutoff=0.8)
masker.fit(haxby.func)
cutoff_mask = masker.mask_.get_data().astype(np.bool)
print "Cutoff mask loss: %i" % np.where(haxby_func[-cutoff_mask] != 0)[0].size

# Plot the masks
pl.subplot(2, 2, 1)
pl.imshow(np.rot90(haxby_mask[..., 27]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.title('Haxby mask')

pl.subplot(2, 2, 2)
pl.imshow(np.rot90(default_mask[..., 27]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.title('Default mask')

pl.subplot(2, 2, 3)
pl.imshow(np.rot90(opening_mask[..., 27]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.title('Mask with opening')

pl.subplot(2, 2, 4)
pl.imshow(np.rot90(cutoff_mask[..., 27]), interpolation='nearest',
        cmap=pl.cm.gray)
pl.title('Mask with cutoff')

pl.show()
