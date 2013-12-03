"""In this script we reproduce the original data analysis conducted by
Haxby et al. in 

"Distributed and Overlapping Representations of Faces and Objects in Ventral Temporal Cortex"

(Science 2001)

"""


### Fetch data using nilearn dataset fetcher ################################
# specify how many subjects we want to load
n_subjects = 2

import os
from nilearn import datasets
data_files = datasets.fetch_haxby(n_subjects=n_subjects, fetch_stimuli=True)
# get root folder from first entry in data files
data_dir = os.path.dirname(os.path.dirname(data_files.anat[0]))


### Take a quick look at stimuli if option is set to True ###################
take_a_look_at_stimuli = False

if take_a_look_at_stimuli:
    import Image
    import pylab as pl
    main_stimuli_dir = os.path.join(data_dir, "stimuli")
    for file_dictionary in data_files.stimuli:
        if "controls" in file_dictionary.keys():
            # skip control images, there are too many
            continue

        stim_type = file_dictionary.keys()[0]
        file_names = file_dictionary[stim_type]
        stim_dir = os.path.join(main_stimuli_dir, stim_type)

        pl.figure()
        for i in range(48):
            pl.subplot(6, 8, i + 1)
            try:
                pl.imshow(Image.open(
                    os.path.join(stim_dir, file_names[i])),
                    origin="lower")

                pl.gray()
                pl.axis("off")
            except:
                # just go to the next one if the file is not present
                continue
        pl.title(stim_type)
    pl.show()

### Do first similarity analysis for all subjects in subjects list ##########
subject_ids = [1]

# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.input_data import NiftiMasker
import numpy as np


# @@@@@ Michael: Maybe we should only work with one subject? This outer loop is easily removed
for subject_id in subject_ids:

    # create masker object
    # @@@@@ Michael: Do we detrend and standardize here? Data loaded goes across 12 runs, so ideally it should be broken apart. Scipy.signal.detrend has this breakpoint option. The paper does normalization across conditions (subtracting the condition mean), but I don't know if it does detrending and standardizing before this. Subtracting timecourse means definitely makes the data look a lot better - one can distinguish rest from activation times. I will do the detrending and standardizing by block later on
    ventral_temporal_mask = NiftiMasker(data_files.mask_vt[subject_id])
                                        # detrend=True,
                                        # standardize=True)
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
    # @@@@@ Michael: It may be useful to teach how this is done explicitly. Then again, the NiftiMasker could probably also be made to be able to do this easily
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
        import pylab as pl
        pl.figure()
        pl.plot(vt_timecourses[:121])
        pl.xlabel("TRs")
        pl.ylabel("Normalized BOLD")
        pl.title("Some BOLD time courses.")

    # load labels
    labels = np.recfromcsv(data_files.session_target[subject_id],
                           delimiter=" ")
    # Possibly take a quick look at them
    print "Label names in dataset:"
    print np.unique(labels['labels'])
    # This outputs:
    # ['bottle' 'cat' 'chair' 'face' 'house' 'rest'
    # 'scissors' 'scrambledpix' 'shoe']

    # Citing the paper:
    """ To identify object-selective cortex, we used an eight-regressor model. The first regressor was the contrast between stimulus blocks and rest. The remaining seven regressors modeled the response to each meaningful category."""
    # Let's do it:
    unique_conditions = np.unique(labels)
    # We first make a design matrix containing a regressor for each
    # condition per run [e.g. ("shoe", 3), ("face", 12) and so on]
    independent_design = (labels[:, np.newaxis]
                  == unique_conditions[np.newaxis, :]).astype(np.float64)

    # now we merge the regressors of the same category over different runs
    unique_categories = np.unique(unique_conditions["labels"])
    merge_table = (unique_conditions["labels"][:, np.newaxis]
                   == unique_categories[np.newaxis, :])

    category_design = independent_design.dot(merge_table)

    # category_design is almost what we want. We need to modify one 
    # regressor, rest to rest_vs_active, and drop the scrambled regressor

    rest_vs_active = (category_design[:, unique_categories == "rest"] -
                      category_design[:, unique_categories != "rest"
                                         ].sum(axis=1)[:, np.newaxis])

    active_categories = np.array([s not in ["rest", "scrambledpix"] 
                                  for s in unique_categories])

    eight_regressor_design = np.hstack([rest_vs_active,
                             category_design[:, active_categories]])

    # Fit a GLM using this matrix
    from sklearn.linear_model import LinearRegression
    glm_eight = LinearRegression(fit_intercept=True)

    glm_eight.fit(eight_regressor_design, vt_timecourses)

    beta_maps = ventral_temporal_mask.inverse_transform(glm_eight.coef_.T)

    # Citing the paper again
    """To determine the patterns of response to each category on even-numbered and odd-numbered runs, we used a 16-regressor model - eight regressors to model the response to  each category relative to rest on even runs and eight regressors to model the response to each category on odd runs with no regressor that contrasteed all stimulus blocks to rest"""
    # As unclear as this paragraph is, let us still try to make this matrix
    even_runs = labels["chunks"] % 2 == 0
    odd_runs = labels["chunks"] % 2 == 1

    even_design = (even_runs[:, np.newaxis] * 
                   independent_design).dot(merge_table)
    odd_design = (odd_runs[:, np.newaxis] * 
                   independent_design).dot(merge_table)

    # now subtract the rest regressor from all the others
    even_category_vs_rest = (even_design[:, unique_categories != "rest"] -
                even_design[:, unique_categories == "rest"])
    odd_category_vs_rest = (odd_design[:, unique_categories != "rest"] -
                odd_design[:, unique_categories == "rest"])

    sixteen_regressor_design = np.hstack([even_category_vs_rest,
                                          odd_category_vs_rest])


    # use this matrix in a GLM
    glm_sixteen = LinearRegression(fit_intercept=True)
    glm_sixteen.fit(sixteen_regressor_design, vt_timecourses)

    activations = glm_sixteen.coef_
    # as indicated in the paper, normalize activations across categories,
    # (but not across even/odd runs)
    normalized_activations = ((activations.reshape(-1, 2, 8)
           - activations.reshape(-1, 2, 8).mean(-1)[..., np.newaxis])
           / activations.reshape(-1, 2, 8).std(-1)[..., np.newaxis]
                              ).reshape(-1, 16)

    correlation_matrix = np.corrcoef(normalized_activations.T)

    plot_labels = list(unique_categories[unique_categories != "rest"])

    import pylab as pl
    pl.figure()
    pl.imshow(correlation_matrix, interpolation="nearest")
    pl.title("Full correlation matrix, \nodd/even correlation on off diagonal blocks")
    pl.yticks(range(16), plot_labels * 2)
    pl.xticks(range(16), plot_labels * 2, rotation=90)
    pl.jet()
    pl.colorbar()
    pl.show()



    # Now try a quick and dirty SVM on the whole thing
    from sklearn.svm import SVC
    # from sklearn.feature_selection import f_classif, SelectKBest
    # from sklearn.pipeline import Pipeline
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.cross_validation import cross_val_score
    # feature_selection = SelectKBest(f_classif, 500)
    classifier = OneVsRestClassifier(SVC(C=1.))

    scores = cross_val_score(classifier, vt_timecourses, labels['labels'],
                             cv=12, n_jobs=12, verbose=True)

    # mean score around .8, chance level is at around .11111
    # Note that I didn't even remove resting state here.
