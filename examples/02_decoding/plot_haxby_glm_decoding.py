"""
Decoding of a dataset after GLM fit for signal extraction
=========================================================

Full step-by-step example of fitting a GLM to perform a decoding experiment.
We use the data from one subject of the Haxby dataset.

More specifically:

1. Download the Haxby dataset.
2. Extract the information to generate a glm
   representing the blocks of stimuli.
3. Analyze the decoding performance using a classifier.
"""

##############################################################################
# Fetch example Haxby dataset
# ----------------------------
# We download the Haxby dataset
# This is a study of visual object category representation

# By default 2nd subject will be fetched
import numpy as np
import pandas as pd
from nilearn import datasets

haxby_dataset = datasets.fetch_haxby()

# repetition has to be known
TR = 2.5

##############################################################################
# Load the behavioral data
# -------------------------

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral["labels"].values

# Record these as an array of sessions
sessions = behavioral["chunks"].values
unique_sessions = behavioral["chunks"].unique()

# fMRI data: a unique file for each session
func_filename = haxby_dataset.func[0]

##############################################################################
# Build a proper event structure for each session
# ------------------------------------------------

events = {}
# events will take  the form of a dictionary of Dataframes, one per session
for session in unique_sessions:
    # get the condition label per session
    conditions_session = conditions[sessions == session]
    # get the number of scans per session, then the corresponding
    # vector of frame times
    n_scans = len(conditions_session)
    frame_times = TR * np.arange(n_scans)
    # each event last the full TR
    duration = TR * np.ones(n_scans)
    # Define the events object
    events_ = pd.DataFrame(
        {
            "onset": frame_times,
            "trial_type": conditions_session,
            "duration": duration,
        }
    )
    # remove the rest condition and insert into the dictionary
    events[session] = events_[events_.trial_type != "rest"]

##############################################################################
# Instantiate and run FirstLevelModel
# ------------------------------------
#
# We generate a list of z-maps together with their session and condition index

z_maps = []
conditions_label = []
session_label = []

# Instantiate the glm
from nilearn.glm.first_level import FirstLevelModel

glm = FirstLevelModel(
    t_r=TR,
    mask_img=haxby_dataset.mask,
    high_pass=0.008,
    smoothing_fwhm=4,
    memory="nilearn_cache",
)

##############################################################################
# Run the glm on data from each session
# --------------------------------------
events[session].trial_type.unique()
from nilearn.image import index_img

for session in unique_sessions:
    # grab the fmri data for that particular session
    fmri_session = index_img(func_filename, sessions == session)

    # fit the glm
    glm.fit(fmri_session, events=events[session])

    # set up contrasts: one per condition
    conditions = events[session].trial_type.unique()
    for condition_ in conditions:
        z_maps.append(glm.compute_contrast(condition_))
        conditions_label.append(condition_)
        session_label.append(session)

##############################################################################
# Generating a report
# --------------------
# Since we have already computed the FirstLevelModel
# and have the contrast, we can quickly create a summary report.

from nilearn.image import mean_img
from nilearn.reporting import make_glm_report

mean_img_ = mean_img(func_filename)
report = make_glm_report(
    glm,
    contrasts=conditions,
    bg_img=mean_img_,
)

report  # This report can be viewed in a notebook

##############################################################################
# In a jupyter notebook, the report will be automatically inserted, as above.
# We have several other ways to access the report:

# report.save_as_html('report.html')
# report.open_in_browser()

##############################################################################
# Build the decoding pipeline
# ----------------------------
# To define the decoding pipeline we use Decoder object, we choose :
#
#     * a prediction model, here a Support Vector Classifier, with a linear
#       kernel
#
#     * the mask to use, here a ventral temporal ROI in the visual cortex
#
#     * although it usually helps to decode better, z-maps time series don't
#       need to be rescaled to a 0 mean, variance of 1 so we use
#       standardize=False.
#
#     * we use univariate feature selection to reduce the dimension of the
#       problem keeping only 5% of voxels which are most informative.
#
#     * a cross-validation scheme, here we use LeaveOneGroupOut
#       cross-validation on the sessions which corresponds to a
#       leave-one-session-out
#
# We fit directly this pipeline on the Niimgs outputs of the GLM, with
# corresponding conditions labels and session labels
# (for the cross validation).

from nilearn.decoding import Decoder
from sklearn.model_selection import LeaveOneGroupOut

decoder = Decoder(
    estimator="svc",
    mask=haxby_dataset.mask,
    standardize=False,
    screening_percentile=5,
    cv=LeaveOneGroupOut(),
)
decoder.fit(z_maps, conditions_label, groups=session_label)

# Return the corresponding mean prediction accuracy compared to chance

classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
chance_level = 1.0 / len(np.unique(conditions))
print(
    "Classification accuracy: {:.4f} / Chance level: {}".format(
        classification_accuracy, chance_level
    )
)
