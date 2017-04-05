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
classifiers = ['svc_l2', 'svc_l1', 'logistic_l1', 'logistic_l2']

#############################################################################
# Here we compute prediction scores and run time for all these
# classifiers

from nilearn.decoding import Decoder
from nilearn.image import index_img
# Because the data is in one single large 4D image, we need to use
# index_img to do the split easily.
fmri_niimgs = index_img(func_filename, task_mask)
classification_target = stimuli[task_mask]

import time
from sklearn.cross_validation import LeaveOneLabelOut
cv = LeaveOneLabelOut(session_labels)

classifiers_scores = {}
for classifier_name in sorted(classifiers):
    classifiers_scores[classifier_name] = {}
    print(70 * '_')

    decoder = Decoder(estimator=classifier_name, mask=mask_filename,
                      standardize=True, scoring='f1', cv=cv)
    t0 = time.time()
    decoder.fit(fmri_niimgs, classification_target)

    classifiers_scores[classifier_name] = decoder.cv_scores_

    for category in categories:
        print("%10s: %14s -- scores: %1.2f +- %1.2f, time %.2fs" % (
            classifier_name, category,
            np.mean(classifiers_scores[classifier_name][category]),
            np.std(classifiers_scores[classifier_name][category]),
            time.time() - t0))

###############################################################################
# Then we make a rudimentary diagram
import matplotlib.pyplot as plt
plt.figure()

tick_position = np.arange(len(categories))
plt.yticks(tick_position + 0.5, categories)

for i, (color, classifier_name) in enumerate(zip(['b', 'm', 'k', 'r'],
                                                 sorted(classifiers))):
    score_means = [np.mean(classifiers_scores[classifier_name][category])
                   for category in categories]
    plt.barh(tick_position, score_means,
             label=classifier_name.replace('_', ' '),
             height=.2, color=color)
    tick_position = tick_position + .2

plt.xlabel('Classification accurancy (f1 score)')
plt.ylabel('Visual stimuli category')
plt.xlim(xmin=0)
plt.legend(loc='lower right', ncol=1)
plt.title('Category-specific classification accuracy for different classifiers')
plt.tight_layout()


###############################################################################
# Finally, w plot the face vs house map for the different classifiers

# Use the average EPI as a background
from nilearn.image import mean_img
mean_epi_img = mean_img(func_filename)

# Restrict the decoding to face vs house
condition_mask = np.logical_or(stimuli == b'face', stimuli == b'house')
stimuli = stimuli[condition_mask]
fmri_niimgs_condition = index_img(func_filename, condition_mask)

from nilearn.plotting import plot_stat_map, show

for classifier_name in sorted(classifiers):

    decoder = Decoder(estimator=classifier_name, mask=mask_filename,
                      standardize=True, scoring='f1')

    decoder.fit(fmri_niimgs_condition, stimuli)

    threshold = np.max(np.abs(decoder.coef_)) * 1e-3
    plot_stat_map(decoder.coef_img_['house'], bg_img=mean_epi_img,
                  display_mode='z', cut_coords=[-15],
                  threshold=threshold,
                  title='%s: face vs house' % classifier_name)

show()
