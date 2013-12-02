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

    # load labels
    labels = np.recfromcsv(data_files.session_target[subject_id],
                           delimiter=" ")
    # Possibly take a quick look at them
    print "Label names in dataset:"
    print np.unique(labels['labels'])
    # This outputs:
    # ['bottle' 'cat' 'chair' 'face' 'house' 'rest'
    # 'scissors' 'scrambledpix' 'shoe']

    # @@@@@@ Michael: If I understand the paper correctly, they work on the BOLD data. But maybe I am wrong. In the following, I will remove resting state by hand and divide into conditions. However, the design is orthogonal, and would almost stay that way even after convolution with an HRF, so maybe the way to go for this tutorial would rather be via Beta maps?
    # find non resting state episodes
    non_rest = labels["labels"] != "rest"

    # The data consists of 12 concatenated runs of 121 timepoints each,
    # and we will reshape the data to reflect this. This is important, since
    # the original paper correlates activations from even and odd runs

    vt_timecourses = vt_timecourses.reshape(12, 121, 
                                            vt_timecourses.shape[-1])

    # thus reshaped we can detrend and standardize the timecourses
    # @@@@@ Michael: It may be useful to teach how this is done explicitly. Then again, the NiftiMasker could probably also be made to be able to do this easily
    from scipy.signal import detrend
    vt_timecourses = detrend(vt_timecourses, axis=1)
    vt_timecourses = ((vt_timecourses
                       - vt_timecourses.mean(axis=1)[:, np.newaxis, :])
                       / vt_timecourses.std(axis=1)[:, np.newaxis, :])

    # We could take a look at some BOLD time courses
    take_a_look_at_bold = True
    if take_a_look_at_bold:
        import pylab as pl
        pl.figure()
        pl.plot(vt_timecourses[0])
        pl.xlabel("TRs")
        pl.ylabel("Normalized BOLD")
        pl.title("Some BOLD time courses.")
    # we reshape the non_rest indicator accordingly
    non_rest = non_rest.reshape(12, 121)

    # and find -- possibly by plotting it ...
    plot_non_rest = False
    if plot_non_rest:
        import pylab as pl
        pl.figure()
        pl.plot(non_rest.T)

    # ... that the rest-non_rest structure is the same over all runs. We can
    # assert the fact that all runs look like the first one like so:
    assert (non_rest == non_rest[0:1]).all()

    # Now remove the resting state data
    active_timecourses = vt_timecourses[:, non_rest[0], :]

    active_labels = labels['labels'][non_rest.ravel()].reshape(12, 72)

    # In printing the active labels, we find that within a run there is
    # always the same block structure of 8 blocks of 9 TRs each, where 
    # each type of stimulus is shown. However: The order is shuffled
    # from run to run.

    print "Active labels"
    print active_labels

    # We exploit the inner block structure of the runs by reshaping our
    # data once more: 12 runs of 8 blocks each consisting of 9 TRs.
    active_timecourses = active_timecourses.reshape(12, 8, 9,
                                          active_timecourses.shape[-1])

    # In reshaping the label array according to this same structure, 
    # we can extract the label pertaining to each block

    block_labels = active_labels.reshape(12, 8, 9)[..., 0]
    # @@@@@ Michael: I am realizing more and more, that I may be doing too much acrobatics with ndarrays. More and more convinced that Beta maps are the way to go.

    # According to the paper, BOLD activity is 
