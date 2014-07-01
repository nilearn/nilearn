"""
S-Lasso Simple example of decoding: the Haxby data
==================================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs house discrimination task in a mask of the ventral
stream.

Author: DOHMATOB Elvis Dopgima

"""

### Load haxby dataset ########################################################

from nilearn import datasets
data_files = datasets.fetch_haxby()

### Load Target labels ########################################################

import numpy as np
# Load target information as string and give a numerical identifier to each
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")
cond1 = "face"
cond2 = "house"
vt_masking = False

from sklearn.externals.joblib import Memory
from nilearn.input_data import NiftiMasker
memory = Memory("cache")

condition_mask = np.logical_or(labels['labels'] == cond1,
                               labels['labels'] == cond2)
_, target = np.unique(labels['labels'], return_inverse=True)

if vt_masking:
    # ventral mask
    nifti_masker = NiftiMasker(mask=data_files.mask_vt[0], standardize=True)
    epi = data_files.func[0]
else:
    # occipital mask
    import nibabel
    epi = nibabel.load(data_files.func[0])

    # only keep back of the brain
    data = epi.get_data()
    data[:, data.shape[1] / 2:, :] = 0
    epi = nibabel.Nifti1Image(data, epi.get_affine())
    nifti_masker = NiftiMasker(standardize=True, memory=memory,
                         mask_strategy='epi')

fmri_masked = nifti_masker.fit_transform(epi)[condition_mask]
target = target[condition_mask]
session_labels = labels['chunks'][condition_mask]

### Cross-validation ##########################################################
# Here we use a SmoothLasso classifier
from nilearn.decoding.sparse_models.cv import SmoothLassoClassifierCV
import os
n_jobs = int(os.environ.get("N_JOBS", 1))
mask = nifti_masker.mask_img_.get_data().astype(np.bool)
slcv = SmoothLassoClassifierCV(l1_ratio=.5, verbose=1, memory=memory,
                               mask=mask, n_jobs=n_jobs
                               ).fit(fmri_masked, target)

### Unmasking #################################################################

# Retrieve the SLCV discriminating weights
coef_ = slcv.coef_

# Reverse masking thanks to the Nifti Masker
coef_niimg = nifti_masker.inverse_transform(coef_)

# Use nibabel to save the coefficients as a Nifti image
import nibabel
nibabel.save(coef_niimg, 'haxby_slcv_weights.nii')

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image.image import mean_img
from nilearn.plotting import plot_stat_map

# weights
mean_epi = mean_img(data_files.func[0])
slicer = plot_stat_map(coef_niimg, mean_epi, title="Smooth Lasso weights")
slicer.contour_map(nifti_masker.mask_img_, levels=[.5], colors='r')

# CV
from nilearn.decoding.sparse_models.cv import plot_cv_scores
plot_cv_scores(slcv)
print ("Accurarcy: %g" % ((slcv.predict(fmri_masked) == target
                           ).mean() * 100.)) + "%"
plt.show()
