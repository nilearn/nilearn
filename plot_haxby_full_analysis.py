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

    # create masker object
    ventral_temporal_mask = NiftiMasker(data_files.mask_vt[subject_id])

    # initialize it
    ventral_temporal_mask.fit()
    # mask the BOLD data
    vt_timecourses = ventral_temporal_mask.transform(
        data_files.func[subject_id])

    # The data consist of 12 concatenated runs of 121 timepoints each,
    # and we will reshape the data to reflect this. This is important, since
    # detrending and standardizing needs to respect this structure

    vt_timecourses = vt_timecourses.reshape(12, 121, 
                                            vt_timecourses.shape[-1])

    # thus reshaped we can detrend and standardize the timecourses
    from scipy.signal import detrend
    vt_timecourses = detrend(vt_timecourses, axis=1)
    vt_timecourses = ((vt_timecourses
                       - vt_timecourses.mean(axis=1)[:, np.newaxis, :])
                       / vt_timecourses.std(axis=1)[:, np.newaxis, :])
    # reshape back
    vt_timecourses = vt_timecourses.reshape(12 * 121, -1)
    # We could take a look at some BOLD time courses
    take_a_look_at_bold = False
    if take_a_look_at_bold:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(vt_timecourses[:121])
        plt.xlabel("TRs")
        plt.ylabel("Normalized BOLD")
        plt.title("Some BOLD time courses.")

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

    scores = cross_val_score(classifier,
                             vt_timecourses[resting_state == False],
                             labels['labels'][resting_state == False],
                             cv=12, n_jobs=1, verbose=True)

    # After removing resting state:
    # mean score around .82 (.86 with RS), chance level is .125
    print "Linear SVM C=1 on ROI"
    print "Score: %1.2f +- %1.2f" % (scores.mean(), scores.std())

    # Now full brain ANOVA + Linear SVM

    # need to extract full brain mask
    from nilearn.masking import compute_epi_mask
    brain_mask = compute_epi_mask(data_files.func[subject_id])
    brain_masker = NiftiMasker(brain_mask)
    all_timecourses = brain_masker.fit_transform(data_files.func[subject_id])
    from sklearn.feature_selection import f_classif, SelectKBest
    from sklearn.pipeline import Pipeline
    feature_selection = SelectKBest(f_classif, 500)
    pipeline = Pipeline([("Feature selection", feature_selection),
                         ("Classifier", classifier)])

    scores_anova_svm = cross_val_score(pipeline,
                                       all_timecourses[resting_state == False], 
                                       labels['labels'][resting_state == False],
                                       cv=12, n_jobs=1,
                                       verbose=True)
    # With resting removed state this scores at .91 (with RS: .87)
    print "ANOVA + Linear SVM C=1 on full brain"
    print "Score: %1.2f +- %1.2f" % (scores_anova_svm.mean(),
                                     scores_anova_svm.std())
