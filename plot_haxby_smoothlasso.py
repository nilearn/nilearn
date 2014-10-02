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

from sklearn.externals.joblib import Memory
from nilearn.input_data import NiftiMasker
memory = Memory("cache")

condition_mask = np.logical_or(labels['labels'] == cond1,
                               labels['labels'] == cond2)
_, target = np.unique(labels['labels'], return_inverse=True)

# ventral mask
import nibabel
nifti_masker = NiftiMasker(mask_img=data_files.mask_vt[0], standardize=True)

# make X (design matrix) and y (dependent variate)
niimgs  = nibabel.load(data_files.func[0])
X = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask],
                        niimgs.get_affine())
y = target[condition_mask]

### Fit S-LASSO classifier and retreive weights map ##########################
from nilearn.decoding import SpaceNet
decoder = SpaceNet(memory=memory, mask=nifti_masker, classif=True, verbose=1,
                penalty="tvl1", n_jobs=14)
decoder.fit(X, y)
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('haxby_slcv_weights.nii')

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image.image import mean_img
from nilearn.plotting import plot_stat_map

mean_epi = mean_img(data_files.func[0])
slicer = plot_stat_map(coef_niimg, mean_epi, title="S-LASSO weights")
print ("Accurarcy: %g" % ((decoder.predict(X) == y).mean() * 100.)) + "%"
print "_" * 80
plt.show()
