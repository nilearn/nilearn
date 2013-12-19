"""
Different classifiers in decoding the Haxby dataset
=====================================================

Here we compare different classifiers on a visual object recognition
decoding task.
"""

import time

### Fetch data using nilearn dataset fetcher ################################
from nilearn import datasets
data_files = datasets.fetch_haxby(n_subjects=1)

# load labels
import numpy as np
labels = np.recfromcsv(data_files.session_target[0], delimiter=" ")
stimuli = labels['labels']
# identify resting state labels in order to be able to remove them
resting_state = stimuli == "rest"

# find names of remaining active labels
categories = np.unique(stimuli[resting_state == False])

# extract tags indicating to which acquisition run a tag belongs
session_labels = labels["chunks"][resting_state == False]

# Load the fMRI data
from nilearn.input_data import NiftiMasker

masker = NiftiMasker(mask=data_files['mask_vt'][0])
masked_timecourses = masker.fit_transform(
    data_files.func[0])[resting_state == False]

### Classifiers definition

# A support vector classifier
from sklearn.svm import SVC
svm = SVC(C=1., kernel="linear")

from sklearn.grid_search import GridSearchCV
# GridSearchCV is slow, but note that it takes an 'n_jobs' parameter that
# can significantly speed up the fitting process on computers with
# multiple cores
svm_cv = GridSearchCV(SVC(C=1., kernel="linear"),
                      param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                      scoring='f1')

# The logistic regression
from sklearn.linear_model import LogisticRegression, RidgeClassifier, \
    RidgeClassifierCV
logistic = LogisticRegression(C=1., penalty="l1")
logistic_50 = LogisticRegression(C=50., penalty="l1")
logistic_l2 = LogisticRegression(C=1., penalty="l2")

logistic_cv = GridSearchCV(LogisticRegression(C=1., penalty="l1"),
                           param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                      scoring='f1')
logistic_l2_cv = GridSearchCV(LogisticRegression(C=1., penalty="l1"),
                           param_grid={'C': [.1, .5, 1., 5., 10., 50., 100.]},
                      scoring='f1')

ridge = RidgeClassifier()
ridge_cv = RidgeClassifierCV()


# Make a data splitting object for cross validation
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
cv = LeaveOneLabelOut(session_labels)

classifiers = {'SVC': svm,
               'SVC cv': svm_cv,
               'log l1': logistic,
               'log l1 50': logistic_50,
               'log l1 cv': logistic_cv,
               'log l2': logistic_l2,
               'log l2 cv': logistic_l2_cv,
               'ridge': ridge,
               'ridge cv': ridge_cv}

classifiers_scores = {}

for classifier_name, classifier in sorted(classifiers.items()):
    classifiers_scores[classifier_name] = {}
    print 70 * '_'

    for category in categories:
        classification_target = stimuli[resting_state == False] == category
        t0 = time.time()
        classifiers_scores[classifier_name][category] = cross_val_score(
            classifier,
            masked_timecourses,
            classification_target,
            cv=cv, scoring="f1")

        print "%10s: %14s -- scores: %1.2f +- %1.2f, time %.2fs" % (
            classifier_name, category,
            classifiers_scores[classifier_name][category].mean(),
            classifiers_scores[classifier_name][category].std(),
            time.time() - t0)

###############################################################################
# make a rudimentary diagram
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
# Plot the face vs house map for the different estimators
import nibabel

# use the average EPI as a background
epi_img = nibabel.load(data_files.func[0])
mean_epi = epi_img.get_data().mean(axis=-1)

# Restrict the decoding to face vs house
condition_mask = np.logical_or(stimuli == 'face', stimuli == 'house')
masked_timecourses = masked_timecourses[condition_mask[resting_state == False]]
stimuli = stimuli[condition_mask]
# Transform the stimuli to binary values
stimuli = (stimuli == 'face').astype(np.int)

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

    plt.figure(figsize=(3, 5))
    plt.imshow(np.rot90(mean_epi[..., 27]), interpolation='nearest',
            cmap=plt.cm.gray)
    vmax = max(-weight_map.min(), weight_map.max())
    plt.imshow(np.rot90(
               np.ma.masked_inside(weight_map[..., 27], -.001*vmax, .001*vmax)),
               interpolation='nearest', vmax=vmax, vmin=-vmax)
    plt.axis('off')
    plt.title('%s: face vs house' % classifier_name)
    plt.tight_layout()

plt.show()



