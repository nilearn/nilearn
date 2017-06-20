"""
Different classifiers in decoding the Haxby dataset
=====================================================

Here we compare different classifiers on a visual object recognition
decoding task.
"""

#############################################################################
# We start by loading the data and applying simple transformations to it
# -----------------------------------------------------------------------

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
# identify resting state labels in order to be able to remove them
resting_state = stimuli == b'rest'

# find names of remaining active labels
categories = np.unique(stimuli[np.logical_not(resting_state)])

# extract tags indicating to which acquisition run a tag belongs
session_labels = labels["chunks"][np.logical_not(resting_state)]

# Load the fMRI data
from nilearn.input_data import NiftiMasker

# For decoding, standardizing is often very important
mask_filename = haxby_dataset.mask_vt[0]
masker = NiftiMasker(mask_img=mask_filename, standardize=True)
func_filename = haxby_dataset.func[0]
masked_timecourses = masker.fit_transform(
    func_filename)[np.logical_not(resting_state)]


#############################################################################
# Then we define the various classifiers that we use
# ---------------------------------------------------
# A support vector classifier
from sklearn.svm import SVC
svm = SVC(C=1., kernel="linear")

# The logistic regression
from sklearn.linear_model import LogisticRegression, RidgeClassifier, \
    RidgeClassifierCV
logistic = LogisticRegression(C=1., penalty="l1")
logistic_50 = LogisticRegression(C=50., penalty="l1")
logistic_l2 = LogisticRegression(C=1., penalty="l2")

# Cross-validated versions of these classifiers
from sklearn.grid_search import GridSearchCV
# GridSearchCV is slow, but note that it takes an 'n_jobs' parameter that
# can significantly speed up the fitting process on computers with
# multiple cores
svm_cv = GridSearchCV(SVC(C=1., kernel="linear"),
                      param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                      scoring='f1', n_jobs=1)

logistic_cv = GridSearchCV(LogisticRegression(C=1., penalty="l1"),
                           param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                           scoring='f1')
logistic_l2_cv = GridSearchCV(LogisticRegression(C=1., penalty="l2"),
                              param_grid={
                                  'C': [.1, .5, 1., 5., 10., 50., 100.]},
                              scoring='f1')

# The ridge classifier has a specific 'CV' object that can set it's
# parameters faster than using a GridSearchCV
ridge = RidgeClassifier()
ridge_cv = RidgeClassifierCV()

# A dictionary, to hold all our classifiers
classifiers = {'SVC': svm,
               'SVC cv': svm_cv,
               'log l1': logistic,
               'log l1 50': logistic_50,
               'log l1 cv': logistic_cv,
               'log l2': logistic_l2,
               'log l2 cv': logistic_l2_cv,
               'ridge': ridge,
               'ridge cv': ridge_cv}


#############################################################################
# Here we compute prediction scores
# ----------------------------------
# Run time for all these classifiers

# Make a data splitting object for cross validation
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
cv = LeaveOneLabelOut(session_labels)

import time

classifiers_scores = {}

for classifier_name, classifier in sorted(classifiers.items()):
    classifiers_scores[classifier_name] = {}
    print(70 * '_')

    for category in categories:
        task_mask = np.logical_not(resting_state)
        classification_target = (stimuli[task_mask] == category)
        t0 = time.time()
        classifiers_scores[classifier_name][category] = cross_val_score(
            classifier,
            masked_timecourses,
            classification_target,
            cv=cv, scoring="f1")

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


###############################################################################
# Finally, w plot the face vs house map for the different classifiers

# Use the average EPI as a background
from nilearn import image
mean_epi_img = image.mean_img(func_filename)

# Restrict the decoding to face vs house
condition_mask = np.logical_or(stimuli == b'face', stimuli == b'house')
masked_timecourses = masked_timecourses[
    condition_mask[np.logical_not(resting_state)]]
stimuli = stimuli[condition_mask]
# Transform the stimuli to binary values
stimuli = (stimuli == b'face').astype(np.int)

from nilearn.plotting import plot_stat_map, show

for classifier_name, classifier in sorted(classifiers.items()):
    classifier.fit(masked_timecourses, stimuli)

    if hasattr(classifier, 'coef_'):
        weights = classifier.coef_[0]
    elif hasattr(classifier, 'best_estimator_'):
        weights = classifier.best_estimator_.coef_[0]
    else:
        continue
    weight_img = masker.inverse_transform(weights)
    weight_map = weight_img.get_data()
    threshold = np.max(np.abs(weight_map)) * 1e-3
    plot_stat_map(weight_img, bg_img=mean_epi_img,
                  display_mode='z', cut_coords=[-15],
                  threshold=threshold,
                  title='%s: face vs house' % classifier_name)

show()
