"""
Simple example of decoding on the Haxby data, using Space-Net prior
===================================================================

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

from nilearn.input_data import NiftiMasker

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

### Fit and predict #########################################################
from nilearn.decoding import SpaceNet
decoder = SpaceNet(memory="cache", mask=nifti_masker, is_classif=True,
                   verbose=1, penalty="tv-l1")
decoder.fit(X, y)  # fit
y_pred = decoder.predict(X)  # predict
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('haxby_tvl1_weights.nii')


### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map

background_img = mean_img(data_files.func[0])
slicer = plot_stat_map(coef_niimg, background_img, title="TV-L1 weights")
print ("Accuracy: %g" % ((y_pred == y).mean() * 100.)) + "%"
print "_" * 80
plt.show()
