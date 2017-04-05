"""
Different classifiers in decoding the Haxby dataset
=====================================================

Here we compare different classifiers on a visual object recognition
decoding task.
"""

#############################################################################
# We start by loading the data and applying simple transformations to it

# Fetch data using nilearn dataset fetcher
from nilearn import datasets
# by default 2nd subject data will be fetched
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print('First subject anatomical nifti image (3D) located is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is located at: %s' %
      haxby_dataset.func[0])

# load labels
import numpy as np
labels = np.recfromcsv(haxby_dataset.session_target[0], delimiter=" ")
stimuli = labels['labels']
# identify resting state (baseline) labels in order to be able to remove them
resting_state = stimuli == b'rest'

# find names of remaining active labels
categories = np.unique(stimuli[np.logical_not(resting_state)])

# extract tags indicating to which acquisition run a tag belongs
session_labels = labels["chunks"][np.logical_not(resting_state)]

# extract the indices of the images corresponding to some condition or task
task_mask = np.logical_not(resting_state)


# Load the fMRI data
from nilearn.input_data import NiftiMasker

# For decoding, standardizing is often very important
mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
func_filename = haxby_dataset.func[0]

#############################################################################
# Then we define the various classifiers that we use
param_grid = np.array([.1, .5, 1., 5., 10., 50., 100.])
classifiers = {
    'svc_l2': {'C': param_grid},
    'svc_l1': {'C': param_grid},
    'logistic_l1': {'C': param_grid},
    'logistic_l2': {'C': param_grid},
    'ridge_classifier': {'alpha': 1. / param_grid},
}

#############################################################################
# Here we compute prediction scores and run time for all these
# classifiers

from nilearn.decoding import Decoder
from nilearn.image import index_img
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily.
fmri_niimgs = index_img(func_filename, task_mask)

import time

classifiers_scores = {}

for classifier_name, param_grid in sorted(classifiers.items()):
    classifiers_scores[classifier_name] = {}
    print(70 * '_')

    decoder = Decoder(estimator=classifier_name, mask=mask_filename,
                      param_grid=param_grid,
                      standardize=True)

    for category in categories:
        classification_target = (stimuli[task_mask] == category)
        t0 = time.time()
        decoder.fit(fmri_niimgs, classification_target)
        classifiers_scores[classifier_name][category] = decoder.cv_scores_

        print("%10s: %14s -- scores: %1.2f +- %1.2f, time %.2fs" % (
            classifier_name, category,
            classifiers_scores[classifier_name][category].mean(),
            classifiers_scores[classifier_name][category].std(),
            time.time() - t0))


###############################################################################
# Then we make a rudimentary diagram
import matplotlib.pyplot as plt
plt.figure()

tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, classifier_name in zip(
        ['b', 'c', 'm', 'g', 'y', 'k', '.5', 'r', '#ffaaaa'],
        sorted(classifiers)):
    score_means = [classifiers_scores[classifier_name][category].mean()
                   for category in categories]
    plt.bar(tick_position, score_means, label=classifier_name,
            width=.11, color=color)
    tick_position = tick_position + .09

plt.ylabel('Classification accurancy (f1 score)')
plt.xlabel('Visual stimuli category')
plt.ylim(ymin=0)
plt.legend(loc='lower center', ncol=3)
plt.title('Category-specific classification accuracy for different classifiers')
plt.tight_layout()


# ###############################################################################
# # Finally, w plot the face vs house map for the different classifiers

# # Use the average EPI as a background
# from nilearn import image
# mean_epi_img = image.mean_img(func_filename)

# # Restrict the decoding to face vs house
# condition_mask = np.logical_or(stimuli == b'face', stimuli == b'house')
# masked_timecourses = masked_timecourses[
#     condition_mask[np.logical_not(resting_state)]]
# stimuli = stimuli[condition_mask]
# # Transform the stimuli to binary values
# stimuli = (stimuli == b'face').astype(np.int)

# from nilearn.plotting import plot_stat_map, show

# for classifier_name, classifier in sorted(classifiers.items()):
#     classifier.fit(masked_timecourses, stimuli)

#     if hasattr(classifier, 'coef_'):
#         weights = classifier.coef_[0]
#     elif hasattr(classifier, 'best_estimator_'):
#         weights = classifier.best_estimator_.coef_[0]
#     else:
#         continue
#     weight_img = masker.inverse_transform(weights)
#     weight_map = weight_img.get_data()
#     threshold = np.max(np.abs(weight_map)) * 1e-3
#     plot_stat_map(weight_img, bg_img=mean_epi_img,
#                   display_mode='z', cut_coords=[-15],
#                   threshold=threshold,
#                   title='%s: face vs house' % classifier_name)

# show()
