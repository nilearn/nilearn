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


###############################################################################
# From already masked data
from nilearn.input_data import NiftiMasker
import nilearn.image as image
from nilearn.plotting import plot_roi, show

# Load Miyawaki dataset
from nilearn import datasets
miyawaki_dataset = datasets.fetch_miyawaki2008()

# print basic information on the dataset
print('First functional nifti image (4D) is located at: %s' %
      miyawaki_dataset.func[0])  # 4D data

miyawaki_filename = miyawaki_dataset.func[0]
miyawaki_mean_img = image.mean_img(miyawaki_filename)

# This time, we can use the NiftiMasker without changing the default mask
# strategy, as the data has already been masked, and thus lies on a
# homogeneous background

masker = NiftiMasker()
masker.fit(miyawaki_filename)

plot_roi(masker.mask_img_, miyawaki_mean_img,
         title="Mask from already masked data")


###############################################################################
# From raw EPI data

# Load ADHD resting-state dataset
dataset = datasets.fetch_adhd(n_subjects=1)
epi_filename = dataset.func[0]

# Restrict to 100 frames to speed up computation
from nilearn.image import index_img
epi_img = index_img(epi_filename, slice(0, 100))

# To display the background
mean_img = image.mean_img(epi_img)


# Simple mask extraction from EPI images
# We need to specify an 'epi' mask_strategy, as this is raw EPI data
masker = NiftiMasker(mask_strategy='epi')
masker.fit(epi_img)
plot_roi(masker.mask_img_, mean_img, title='EPI automatic mask')

# Generate mask with strong opening
masker = NiftiMasker(mask_strategy='epi', mask_args=dict(opening=10))
masker.fit(epi_img)
plot_roi(masker.mask_img_, mean_img, title='EPI Mask with strong opening')

# Generate mask with a high lower cutoff
masker = NiftiMasker(mask_strategy='epi',
                     mask_args=dict(upper_cutoff=.9, lower_cutoff=.8,
                                    opening=False))
masker.fit(epi_img)
plot_roi(masker.mask_img_, mean_img,
         title='EPI Mask: high lower_cutoff')

###############################################################################
# Extract time series

# trended vs detrended
trended = NiftiMasker(mask_strategy='epi')
detrended = NiftiMasker(mask_strategy='epi', detrend=True)
trended_data = trended.fit_transform(epi_img)
detrended_data = detrended.fit_transform(epi_img)

# The timeseries are numpy arrays, so we can manipulate them with numpy
import numpy as np

print("Trended: mean %.2f, std %.2f" %
      (np.mean(trended_data), np.std(trended_data)))
print("Detrended: mean %.2f, std %.2f" %
      (np.mean(detrended_data), np.std(detrended_data)))

show()
