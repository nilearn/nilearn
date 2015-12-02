"""
Decoding with decoder metaestimator: face vs house recognition
==============================================================

Here is a simple example of decoding with a the decoder metaestimator,
reproducing the Haxby 2001 study on a face vs house discrimination task.
"""

#########################################################################
# Load Haxby dataset
import numpy as np
import nibabel
from nilearn import datasets
from nilearn.image import new_img_like

# Load the Haxby dataset
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby(n_subjects=1)

func_filenames = data_files.func[0]
labels_filenames = data_files.session_target[0]

from nilearn.image import mean_img
background_img = mean_img(func_filenames)

# Load Target labels
import numpy as np
labels = np.recfromcsv(labels_filenames, delimiter=" ")

# Restrict to face and house conditions
target = labels['labels']
condition_mask = np.logical_or(target == "face", target == "house")

# Split data into train and test samples, using the chunks
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 6)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 6)

### Prediction with Decoder ###################################################
from nilearn.decoding import Decoder
decoder = Decoder(estimator='svc_l2', screening_percentile=20, cv=3,
                  mask_strategy='epi', standardize=True, smoothing_fwhm=4,
                  n_jobs=1)

### Fit and predict
decoder.fit(func_filenames, target, index=condition_mask_train)
y_pred = decoder.predict(func_filenames, index=condition_mask_test)

weight_img = decoder.coef_img_['house']
prediction_accuracy = np.mean(decoder.cv_scores_)

print("=== DECODER ===")
print("Prediction accuracy: %f" % prediction_accuracy)
print("")

from nilearn.plotting import plot_stat_map, show
plot_stat_map(weight_img, background_img, cut_coords=(-34, -16),
              display_mode="yz")

show()
