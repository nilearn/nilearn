"""
Automatic mask computation with parameter tweaking
==================================================

In this example, the Nifti masker is used to automatically compute a mask.
Using some visualization, one can see that the default parameters of the
nifti masker are not suited for this dataset. They are consequently tweaked
to obtain a decent mask.
"""

import matplotlib.pyplot as plt
import numpy as np

import nibabel
from nilearn import datasets

# Load Haxby dataset
haxby = datasets.fetch_haxby_simple()
haxby_img = nibabel.load(haxby.func)
# Restrict haxby to 150 frames to speed up computation
haxby_func = haxby_img.get_data()[..., :150]

# haxby_func is a 4D-array, we want to make a Niimg out of it:
haxby_img = nibabel.Nifti1Image(haxby_func, haxby_img.get_affine())

# Display helper
background = np.mean(haxby_func, axis=-1)[..., 27]


def display_mask(background, mask, title):
    plt.axis('off')
    plt.imshow(np.rot90(background), interpolation='nearest', cmap=plt.cm.gray)
    ma = np.ma.masked_equal(mask, False)
    plt.imshow(np.rot90(ma), interpolation='nearest',
              cmap=plt.cm.autumn, alpha=0.5)
    plt.title(title)

# Generate mask with default parameters
from nilearn.input_data import NiftiMasker
masker = NiftiMasker()
masker.fit(haxby_img)
default_mask = masker.mask_img_.get_data().astype(np.bool)
plt.figure(figsize=(3, 5))
display_mask(background, default_mask[..., 27], 'Default mask')

# Generate mask with opening
masker = NiftiMasker(mask_opening=0)
masker.fit(haxby_img)
opening_mask = masker.mask_img_.get_data().astype(np.bool)
plt.figure(figsize=(3, 5))
display_mask(background, opening_mask[..., 27], 'Mask without opening')

# Generate mask with upper cutoff
masker = NiftiMasker(mask_opening=True, mask_upper_cutoff=0.8)
masker.fit(haxby_img)
cutoff_mask = masker.mask_img_.get_data().astype(np.bool)

# Plot the mask and compare it to original

# Load mask provided by Haxby
haxby_mask = nibabel.load(haxby.mask).get_data().astype(np.bool)

plt.figure(figsize=(6, 5))
plt.subplot(1, 2, 1)
display_mask(background, haxby_mask[..., 27], 'Haxby mask')

plt.subplot(1, 2, 2)
display_mask(background, cutoff_mask[..., 27], 'Mask with cutoff')
plt.subplots_adjust(top=0.8)
plt.show()

# trended vs detrended
trended = NiftiMasker(mask=haxby.mask)
detrended = NiftiMasker(mask=haxby.mask, detrend=True)
trended_data = trended.fit_transform(haxby_img)
detrended_data = detrended.fit_transform(haxby_img)

print "Trended: mean %.2f, std %.2f" % \
    (np.mean(trended_data), np.std(trended_data))
print "Detrended: mean %.2f, std %.2f" % \
    (np.mean(detrended_data), np.std(detrended_data))
