"""
Simple example of Nifti Masker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nisl import datasets
from nisl.io import NiftiMasker
dataset = datasets.fetch_haxby()

### Preprocess data ###########################################################

import numpy as np

# Load target information as string and give a numerical identifier to each
labels = np.loadtxt(dataset.session_target[0], dtype=np.str, skiprows=1,
                    usecols=(0,))
index, target = np.unique(labels, return_inverse=True)

nifti_masker = NiftiMasker(mask=dataset.mask_vt[0])
nifti_masker.fit(dataset.func[0])
fmri_masked = nifti_masker.transform(dataset.func[0])

# Remove resting state condition
no_rest_indices = (labels != 'rest')
target = target[no_rest_indices]
fmri_masked = fmri_masked[no_rest_indices]

### Prediction function #######################################################

### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel and C=1
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1.)

svc.fit(fmri_masked, target)
y_pred = svc.predict(fmri_masked)

### Visualisation #############################################################

### Look at the discriminating weights
sv = svc.support_vectors_
# reverse masking
niimg = nifti_masker.inverse_transform(sv[0])

# We use a masked array so that the voxels at '-1' are displayed
# transparently
act = np.ma.masked_array(niimg.get_data(), niimg.get_data() == 0)

### Create the figure
import pylab as pl
pl.axis('off')
pl.title('SVM vectors')
pl.imshow(np.rot90(act[..., 27]), cmap=pl.cm.hot,
          interpolation='nearest')
pl.show()
