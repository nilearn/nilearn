"""
Decoding with SpaceNet on Haxby dataset
==========================================

Here is a simple example of decoding with a SpaceNet prior (i.e S-LASSO,
TV-l1, etc.), reproducing the Haxby 2001 study on a face vs house
discrimination task.
"""
# author: DOHMATOB Elvis Dopgima,
#         VAROQUAUX Gael


### Load haxby dataset ########################################################
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby()

### Load Target labels ########################################################
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")


### split data into train and test samples ####################################
target = labels['labels']
condition_mask = np.logical_or(target == "face", target == "house")
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 6)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 6)

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
penalties = ['smooth-lasso', 'tv-l1']
decoders = {}
accuracies = {}
for penalty in penalties:
    ### Fit model on train data and predict on test data ######################
    decoder = SpaceNetClassifier(memory="cache", penalty=penalty, verbose=2)
    decoder.fit(X_train, y_train)
    y_pred = decoder.predict(X_test)
    accuracies[penalty] = (y_pred == y_test).mean() * 100.
    decoders[penalty] = decoder

### Visualization #############################################################
import matplotlib.pyplot as plt
from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map
background_img = mean_img(data_files.func[0])
print "Results"
print "=" * 80
for penalty, decoder in sorted(decoders.items()):
    coef_img = decoder.coef_img_
    plot_stat_map(coef_img, background_img,
                  title="%s: accuracy %g%%" % (penalty, accuracies[penalty]),
                  cut_coords=(20, -34, -16))
    coef_img.to_filename('haxby_%s_weights.nii' % penalty)
    print "- %s %s" % (penalty, '-' * 60)
    print "Number of train samples : %i" % condition_mask_train.sum()
    print "Number of test samples  : %i" % condition_mask_test.sum()
    print "Classification accuracy : %g%%" % accuracies[penalty]
    print "_" * 80
plt.show()
