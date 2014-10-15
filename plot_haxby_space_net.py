"""
Simple example of decoding on the Haxby data, using Space-Net prior
===================================================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs house discrimination task in a mask of the ventral
stream.

Author: DOHMATOB Elvis Dopgima,
        Gael VAROQUAUX

"""

### Load haxby dataset ########################################################
from nilearn import datasets
data_files = datasets.fetch_haxby()

### Load Target labels ########################################################
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


### split data into train and test samples ####################################
condition_mask = np.logical_or(labels['labels'] == "face",
                               labels['labels'] == "house")
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 9)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 9)

_, target = np.unique(labels['labels'], return_inverse=True)

# make X (design matrix) and y (dependent variate)
import nibabel
niimgs  = nibabel.load(data_files.func[0])
X_train = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_train],
                        niimgs.get_affine())
y_train = target[condition_mask_train]
X_test = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_test],
                        niimgs.get_affine())
y_test = target[condition_mask_test]


### Fit and predict ##########################################################
from nilearn.decoding import SpaceNet
decoder = SpaceNet(memory="cache", is_classif=True, penalty="tv-l1",
                   verbose=2)
decoder.fit(X_train, y_train)  # fit
y_pred = decoder.predict(X_test)  # predict
coef_niimg = decoder.coef_img_
coef_niimg.to_filename('haxby_tvl1_weights.nii')


### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(data_files.func[0])
slicer = plot_stat_map(coef_niimg, background_img, title="TV-L1 weights")
print ("Accuracy: %g" % ((y_pred == y_test).mean() * 100.)) + "%"
print "_" * 80
plt.show()
