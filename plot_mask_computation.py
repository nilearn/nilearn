"""
Understanding NiftiMasker and mask computation
==================================================

In this example, the Nifti masker is used to automatically compute a mask.

For data that has already been masked, the default strategy works out of
the box.

However, for raw EPI, as in resting-state time series, we need to use the
'epi' strategy of the NiftiMasker.

In addition, we show here how to tweak the different parameters of the
underlying mask extraction routine
:func:`nilearn.masking.compute_epi_mask`.

"""

import matplotlib.pyplot as plt
import numpy as np

import nibabel
from nilearn import datasets


###############################################################################
# Simple visualization helper

def display_mask(background, mask, title):
    plt.axis('off')
    plt.imshow(np.rot90(background), interpolation='nearest', cmap=plt.cm.gray)
    ma = np.ma.masked_equal(mask, False)
    plt.imshow(np.rot90(ma), interpolation='nearest',
              cmap=plt.cm.autumn, alpha=0.5)
    plt.title(title)


###############################################################################
# From already masked data
from nilearn.input_data import NiftiMasker

# Load Miyawaki dataset
miyawaki = datasets.fetch_miyawaki2008()
miyawaki_img = nibabel.load(miyawaki.func[0])
miyawaki_func = miyawaki_img.get_data()

background = np.mean(miyawaki_func, axis=-1)[..., 14]

# This time, we can use the NiftiMasker without changing the default mask
# strategy, as the data has already been masked, and thus lies on a
# homogeneous background

masker = NiftiMasker()
masker.fit(miyawaki_img)
default_mask = masker.mask_img_.get_data().astype(np.bool)
plt.figure(figsize=(4, 4.5))
display_mask(background, default_mask[..., 14], 'Default background mask')
plt.tight_layout()


###############################################################################
# From raw EPI data

# Load NYU resting-state dataset
nyu = datasets.fetch_nyu_rest(n_subjects=1)
nyu_img = nibabel.load(nyu.func[0])
# Restrict nyu to 100 frames to speed up computation
nyu_func = nyu_img.get_data()[..., :100]

# nyu_func is a 4D-array, we want to make a Niimg out of it:
nyu_img = nibabel.Nifti1Image(nyu_func, nyu_img.get_affine())

# To display the background
background = np.mean(nyu_func, axis=-1)[..., 21]


# Simple mask extraction from EPI images
from nilearn.input_data import NiftiMasker
# We need to specify an 'epi' mask_strategy, as this is raw EPI data
masker = NiftiMasker(mask_strategy='epi')
masker.fit(nyu_img)
default_mask = masker.mask_img_.get_data().astype(np.bool)
plt.figure(figsize=(4, 4.5))
display_mask(background, default_mask[..., 21], 'EPI automatic mask')
plt.tight_layout()

# Generate mask with strong opening
masker = NiftiMasker(mask_strategy='epi', mask_args=dict(opening=10))
masker.fit(nyu_img)
opening_mask = masker.mask_img_.get_data().astype(np.bool)
plt.figure(figsize=(4, 4.5))
display_mask(background, opening_mask[..., 21], 'EPI Mask with strong opening')
plt.tight_layout()

# Generate mask with a high lower cutoff
masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(upper_cutoff=.9, lower_cutoff=.8,
                                    opening=False))
masker.fit(nyu_img)
cutoff_mask = masker.mask_img_.get_data().astype(np.bool)

plt.figure(figsize=(4, 4.5))
display_mask(background, cutoff_mask[..., 21], 'EPI Mask: high lower_cutoff')
plt.tight_layout()

################################################################################
# Extract time series

# trended vs detrended
trended = NiftiMasker(mask_strategy='epi')
detrended = NiftiMasker(mask_strategy='epi', detrend=True)
trended_data = trended.fit_transform(nyu_img)
detrended_data = detrended.fit_transform(nyu_img)

print "Trended: mean %.2f, std %.2f" % \
    (np.mean(trended_data), np.std(trended_data))
print "Detrended: mean %.2f, std %.2f" % \
    (np.mean(detrended_data), np.std(detrended_data))


plt.show()
