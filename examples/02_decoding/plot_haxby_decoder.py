"""
Decoding with decoder metaestimator: face vs house recognition
==============================================================

Here is a simple example of decoding with the decoder metaestimator,
reproducing a face vs house discrimination task on the study Haxby 2001.

    * J.V. Haxby et al. "Distributed and Overlapping Representations of Faces
      and Objects in Ventral Temporal Cortex", Science vol 293 (2001),
      p 2425.-2430.

"""

# Load Haxby dataset
from nilearn.datasets import fetch_haxby
data_files = fetch_haxby()

func_filenames = data_files.func[0]
labels_filenames = data_files.session_target[0]

from nilearn.image import mean_img
background_img = mean_img(func_filenames)

# Load Target labels
import numpy as np
labels = np.recfromcsv(labels_filenames, delimiter=" ")

# Restrict to face and house conditions
target = labels['labels']
condition_mask = np.logical_or(target == b"face", target == b"house")
# condition_mask = np.logical_or(
#     target == b"bottle", np.logical_or(target == b"face", target == b"house"))

# Split data into train and test samples, using the chunks
condition_mask_train = np.logical_and(condition_mask, labels['chunks'] <= 6)
condition_mask_test = np.logical_and(condition_mask, labels['chunks'] > 6)


# Apply this sample mask to X (fMRI data) and y (behavioral labels)
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily
from nilearn.image import index_img

func_filenames = data_files.func[0]
X_train = index_img(func_filenames, condition_mask_train)
X_test = index_img(func_filenames, condition_mask_test)
y_train = target[condition_mask_train]
y_test = target[condition_mask_test]

# Fit and predict with the decoder
from nilearn.decoding import Decoder
decoder = Decoder(estimator='svc_l2', cv=10, mask_strategy='epi',
                  smoothing_fwhm=4, scoring='accuracy', n_jobs=1)

decoder.fit(X_train, y_train)
accuracy = np.mean(decoder.cv_scores_)
print("Decoder cross-validation accuracy : %g%%" % accuracy)

# Testing on out-of-sample data
y_pred = decoder.predict(X_test)
accuracy = (y_pred == y_test).mean() * 100.
print("Decoder classification accuracy : %g%%" % accuracy)

weight_img = decoder.coef_img_[b"face"]

from nilearn.plotting import plot_stat_map, show
plot_stat_map(weight_img, background_img, cut_coords=[-52, -5],
              display_mode="yz",
              title="Decoder: accuracy %g%%" % accuracy)

show()
