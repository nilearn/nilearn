"""
Simple example of decoding on the Haxby data, using Space-Net prior
===================================================================

Here is a simple example of decoding, reproducing the Haxby 2001
study on a face vs house discrimination task in a mask of the ventral
stream.

Author: DOHMATOB Elvis Dopgima,
        VAROQUAUX Gael

"""

### Load haxby dataset ########################################################
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby()

### Load Target labels ########################################################
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


### split data into train and test samples ####################################
target = labels['labels']
condition_mask = np.logical_or(target == "face", target == "house")
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 9)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 9)

# make X (design matrix) and y (response variable)
import nibabel
niimgs  = nibabel.load(data_files.func[0])
X_train = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_train],
                        niimgs.get_affine())
y_train = target[condition_mask_train]
X_test = nibabel.Nifti1Image(niimgs.get_data()[:, :, :, condition_mask_test],
                        niimgs.get_affine())
y_test = target[condition_mask_test]


### Loop over Smooth-LASSO and TV-L1 penalties ###############################
from nilearn.decoding import SpaceNetClassifier
penalties = ['Smooth-LASSO', 'TV-L1']
decoders = {}
accuracies = {}
for penalty in penalties:
   ### Fit model on train data and predict on test data ######################
    decoder = SpaceNetClassifier(memory="cache", penalty=penalty,
                                 verbose=2, max_iter=100)
    decoder.fit(X_train, y_train)  # fit
    y_pred = decoder.predict(X_test)  # predict
    accuracies[penalty] = (y_pred == y_test).mean() * 100.
    decoders[penalty] = decoder

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(data_files.func[0])
print "Results"
print "=" * 80
for penalty, decoder in decoders.iteritems():
    coef_img = decoder.coef_img_
    slicer = plot_stat_map(coef_img, background_img, title=penalty,
                           cut_coords=(20, -34, -16))
    coef_img.to_filename('haxby_%s_weights.nii' % penalty)
    print decoder
    print "#" * 80
    print "Number of train samples : %i" % condition_mask_train.sum()
    print "Number of test samples  : %i" % condition_mask_test.sum()
    print "Classification accuracy : %g%%" % accuracies[penalty]
    print "_" * 80
plt.show()
