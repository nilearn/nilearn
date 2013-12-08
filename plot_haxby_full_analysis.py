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

    # # create masker object
    # ventral_temporal_mask = NiftiMasker(data_files.mask_vt[subject_id])

    # # initialize it
    # ventral_temporal_mask.fit()
    # # mask the BOLD data
    # vt_timecourses = ventral_temporal_mask.transform(
    #     data_files.func[subject_id])

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

    # scores = cross_val_score(classifier,
    #                          vt_timecourses[resting_state == False],
    #                          labels['labels'][resting_state == False],
    #                          cv=12, n_jobs=1, verbose=True)

    # # After removing resting state:
    # # mean score around .82 (.86 with RS), chance level is .125
    # print "Linear SVM C=1 on ROI"
    # print "Score: %1.2f +- %1.2f" % (scores.mean(), scores.std())

    # # Now full brain ANOVA + Linear SVM

    # # need to extract full brain mask
    # from nilearn.masking import compute_epi_mask
    # brain_mask = compute_epi_mask(data_files.func[subject_id])
    # brain_masker = NiftiMasker(brain_mask)
    # all_timecourses = brain_masker.fit_transform(data_files.func[subject_id])
    # from sklearn.feature_selection import f_classif, SelectKBest
    # from sklearn.pipeline import Pipeline
    # feature_selection = SelectKBest(f_classif, 500)
    # pipeline = Pipeline([("Feature selection", feature_selection),
    #                      ("Classifier", classifier)])

    # scores_anova_svm = cross_val_score(pipeline,
    #                                    all_timecourses[resting_state == False], 
    #                                    labels['labels'][resting_state == False],
    #                                    cv=12, n_jobs=1,
    #                                    verbose=True)
    # # With resting removed state this scores at .91 (with RS: .87)
    # print "ANOVA + Linear SVM C=1 on full brain"
    # print "Score: %1.2f +- %1.2f" % (scores_anova_svm.mean(),
    #                                  scores_anova_svm.std())

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

        # mask_scores[mask_name] = cross_val_score(classifier,
        #                      masked_timecourses[resting_state == False],
        #                      labels['labels'][resting_state == False],
        #                      cv=cv, n_jobs=1, verbose=True)
        # print "Scores: %1.2f +- %1.2f" % (mask_scores[mask_name].mean(),
        #                                   mask_scores[mask_name].std())
