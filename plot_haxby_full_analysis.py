"""
ROI based SVM analysis in Haxby et al. dataset
===============================================================================

In this script we reproduce the original data analysis conducted by
Haxby et al. in "Distributed and Overlapping Representations of Faces and
    Objects in Ventral Temporal Cortex"

(Science 2001)
"""


### Fetch data using nilearn dataset fetcher ################################
# specify how many subjects we want to load
n_subjects = 2

from nilearn import datasets
data_files = datasets.fetch_haxby(n_subjects=n_subjects, fetch_stimuli=True)

### Do first similarity analysis for all subjects in subjects list ##########
subject_ids = [1]

# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.input_data import NiftiMasker
import numpy as np


for subject_id in subject_ids:

    # load labels
    labels = np.recfromcsv(data_files.session_target[subject_id],
                           delimiter=" ")
    # identify resting state labels in order to be able to remove them
    resting_state = labels['labels'] == "rest"

    # find names of remaining active labels
    unique_labels = np.unique(labels['labels'][resting_state == False])

    # extract tags indicating to which acquisition run a tag belongs
    session_labels = labels["chunks"][resting_state == False]


    ### Let us now check the provided masks and do decoding on them

    # Make a data splitting object for cross validation
    from sklearn.cross_validation import LeaveOneLabelOut
    cv = LeaveOneLabelOut(session_labels)

    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.cross_validation import cross_val_score
    classifier = OneVsRestClassifier(SVC(C=.1, kernel="linear"))


    mask_names = ['mask_vt', 'mask_face', 'mask_face_little',
                  'mask_house', 'mask_house_little']

    mask_scores = {}
    for mask_name in mask_names:
        print "Working on mask %s" % mask_name
        masker = NiftiMasker(data_files[mask_name][subject_id])
        masked_timecourses = masker.fit_transform(
            data_files.func[subject_id])[resting_state == False]

        mask_scores[mask_name] = {}

        for label in unique_labels:
            print "Treating %s %s" % (mask_name, label)
            classification_target = \
                labels['labels'][resting_state == False] == label
            mask_scores[mask_name][label] = cross_val_score(
                classifier, 
                masked_timecourses,
                classification_target,
                cv=cv,
                n_jobs=1,
                verbose=True,
                scoring="f1")

            print "Scores: %1.2f +- %1.2f" % (
                mask_scores[mask_name][label].mean(),
                mask_scores[mask_name][label].std())

    # make a rudimentary diagram
    import matplotlib.pyplot as plt
    score_means = np.array([[mask_scores[mask_name][label].mean()
                for label in unique_labels] 
                for mask_name in mask_names])
    plt.figure(figsize=(10, 6))
    plt.imshow(score_means, interpolation="nearest")
    plt.xticks(np.arange(len(unique_labels)) + 1, unique_labels, rotation=45)
    plt.yticks(np.arange(len(mask_names)), mask_names, rotation=45)
    plt.colorbar()
    plt.hot()
    plt.show()


