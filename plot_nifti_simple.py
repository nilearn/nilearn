"""
Simple example of Nifti Masker use
"""

### Load nyu_rest dataset #####################################################

from nisl import datasets, utils
from nisl.io import NiftiMasker
dataset = datasets.fetch_nyu_rest(n_subjects=1)
nifti_masker = NiftiMasker()
nifti_masker.fit(dataset.func[0])
mask = nifti_masker.mask_.get_data()

### Visualize the mask ########################################################
import pylab as pl
import numpy as np
import nibabel
pl.figure()
pl.axis('off')
pl.imshow(np.rot90(nibabel.load(dataset.func[0]).get_data()[..., 20, 0]),
        interpolation='nearest', cmap=pl.cm.gray)
ma = np.ma.masked_equal(mask, False)
pl.imshow(np.rot90(ma[..., 20]), interpolation='nearest', cmap=pl.cm.autumn, alpha=0.5)
pl.title("Mask")
pl.show()

# Preprocessed vs Original data
fmri_masked = nifti_masker.fit_transform(dataset.func[0])

pl.figure()
pl.subplot(1, 2, 1)
pl.axis('off')
pl.title('original')
original = utils.check_niimg(dataset.func[0]).get_data()[..., 20, 0]
pl.imshow(np.rot90(original), interpolation='nearest', cmap=pl.cm.gray)

pl.subplot(1, 2, 2)
pl.axis('off')
pl.title('preprocessed')
pp = nifti_masker.inverse_transform(fmri_masked).get_data()[..., 20, 0]
pp[np.where(pp == 0)] = np.min(pp) - 1
pl.imshow(np.rot90(pp), interpolation='nearest', cmap=pl.cm.gray)

pl.show()
