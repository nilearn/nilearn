"""In this script we reproduce the original data analysis conducted by
Haxby et al. in 

"Distributed and Overlapping Representations of Faces and Objects
    in Ventral Temporal Cortex"

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
    # Possibly take a quick look at them
    print "Label names in dataset:"
    print np.unique(labels['labels'])
    # This outputs:
    # ['bottle' 'cat' 'chair' 'face' 'house' 'rest'
    # 'scissors' 'scrambledpix' 'shoe']

    # identify resting state labels in order to be able to remove them
    resting_state = labels['labels'] == "rest"

    # Now try a quick and dirty SVM on the ROI
    from sklearn.svm import SVC
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.cross_validation import cross_val_score
    classifier = OneVsRestClassifier(SVC(C=1., kernel="linear"))

    ### Let us now check the other provided masks and do decoding
    # in the spirit of the original article

    # Make a data splitting object for cross validation
    session_labels = labels["chunks"][resting_state == False]
    from sklearn.cross_validation import LeaveOneLabelOut
    cv = LeaveOneLabelOut(session_labels)

    import sklearn.metrics
    f1_scorer = sklearn.metrics.SCORERS['f1']

    unique_labels = np.unique(labels['labels'][resting_state == False])

    mask_names = ['mask_vt', 'mask_face', 'mask_face_little',
                  'mask_house', 'mask_house_little']

    mask_scores = {}
    for mask_name in mask_names:
        print "Working on mask %s" % mask_name
        masker = NiftiMasker(data_files[mask_name][subject_id])
        masked_timecourses = masker.fit_transform(
            data_files.func[subject_id])

        mask_scores[mask_name] = {}

        for label in unique_labels:
            print "Treating %s %s" % (mask_name, label)
            mask_scores[mask_name][label] = cross_val_score(classifier, 
                  masked_timecourses[resting_state == False],
                  labels['labels'][resting_state == False] == label,
                                                            cv=cv,
                                                            n_jobs=1,
                                                            verbose=True,
                                                            scoring=f1_scorer)

            print "Scores: %1.2f +- %1.2f" % (
                mask_scores[mask_name][label].mean(),
                mask_scores[mask_name][label].std())

