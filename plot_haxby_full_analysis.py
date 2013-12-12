"""
ROI-based decoding analysis in Haxby et al. dataset
=====================================================

In this script we reproduce the data analysis conducted by
Haxby et al. in "Distributed and Overlapping Representations of Faces and
Objects in Ventral Temporal Cortex".

Specifically, we look at decoding accuracy for different objects in
three different masks: the full ventral stream (mask_vt), the house
selective areas (mask_house) and the face selective areas (mask_face),
that have been defined via a standard GLM-based analysis.

"""


### Fetch data using nilearn dataset fetcher ################################
from nilearn import datasets
data_files = datasets.fetch_haxby(n_subjects=1)

# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.input_data import NiftiMasker


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


from sklearn.svm import SVC
classifier = SVC(C=1., kernel="linear")


# Make a data splitting object for cross validation
from sklearn.cross_validation import LeaveOneLabelOut, cross_val_score
cv = LeaveOneLabelOut(session_labels)

mask_names = ['mask_vt', 'mask_face', 'mask_house']

mask_scores = {}
for mask_name in mask_names:
    print "Working on mask %s" % mask_name
    masker = NiftiMasker(mask=data_files[mask_name][0])
    masked_timecourses = masker.fit_transform(
        data_files.func[0])[resting_state == False]

    mask_scores[mask_name] = {}

    for category in categories:
        print "Processing %s %s" % (mask_name, category)
        classification_target = stimuli[resting_state == False] == category
        mask_scores[mask_name][category] = cross_val_score(
            classifier,
            masked_timecourses,
            classification_target,
            cv=cv, scoring="f1")

        print "Scores: %1.2f +- %1.2f" % (
            mask_scores[mask_name][category].mean(),
            mask_scores[mask_name][category].std())

# make a rudimentary diagram
import matplotlib.pyplot as plt
plt.figure()

tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, (mask_name, mask_score) in zip('rgb', mask_scores.items()):
    score_means = [mask_score[category].mean()
                for category in categories]
    plt.bar(tick_position, score_means, label=mask_name,
            width=.25, color=color)
    tick_position = tick_position + .2

plt.ylabel('Classification accurancy (f1 score)')
plt.xlabel('Visual stimuli category')
plt.legend(loc='best')
plt.title('Category-specific classification accuracy for different masks')
plt.tight_layout()


plt.show()

