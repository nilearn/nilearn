"""
Decoding of a dataset after GLM fit for signal extraction
=========================================================

Full step-by-step example of fitting a :term:`GLM`
to perform a decoding experiment. In this decoding analysis,
we will be doing a one-vs-all classification.
We use the data from one subject of the Haxby dataset.

More specifically:

1. Download the Haxby dataset.
2. Extract the information to generate a glm
   representing the blocks of stimuli.
3. Analyze the decoding performance using a classifier.
"""

# %%
# Fetch example Haxby dataset
# ---------------------------
# We download the Haxby dataset
# This is a study of visual object category representation

# By default 2nd subject will be fetched
import numpy as np
import pandas as pd

from nilearn.datasets import fetch_haxby

haxby_dataset = fetch_haxby()

# repetition has to be known
t_r = 2.5

# %%
# Load the behavioral data
# ------------------------

# Load target information as string and give a numerical identifier to each
behavioral = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
conditions = behavioral["labels"].to_numpy()

# Record these as an array of runs
runs = behavioral["chunks"].to_numpy()
unique_runs = behavioral["chunks"].unique()

# fMRI data: a unique file for each run
func_filename = haxby_dataset.func[0]

# %%
# Build a proper event structure for each run
# -------------------------------------------

events = {}
# events will take the form of a dictionary of Dataframes, one per run
for run in unique_runs:
    # get the condition label per run
    conditions_run = conditions[runs == run]
    # get the number of scans per run, then the corresponding
    # vector of frame times
    n_scans = len(conditions_run)
    frame_times = t_r * np.arange(n_scans)
    # each event last the full TR
    duration = t_r * np.ones(n_scans)
    # Define the events object
    events_ = pd.DataFrame(
        {
            "onset": frame_times,
            "trial_type": conditions_run,
            "duration": duration,
        }
    )
    # remove the rest condition and insert into the dictionary
    events[run] = events_[events_.trial_type != "rest"]

# %%
# Instantiate and run FirstLevelModel
# -----------------------------------
#
# We generate a list of z-maps together with their run and condition index

z_maps = []
conditions_label = []
run_label = []

# Instantiate the glm
from nilearn.glm.first_level import FirstLevelModel

glm = FirstLevelModel(
    t_r=t_r,
    mask_img=haxby_dataset.mask,
    high_pass=0.008,
    smoothing_fwhm=4,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

# %%
# Run the :term:`GLM` on data from each run
# -----------------------------------------
events[run].trial_type.unique()
from nilearn.image import index_img

for run in unique_runs:
    # grab the fmri data for that particular run
    fmri_run = index_img(func_filename, runs == run)

    # fit the GLM
    glm.fit(fmri_run, events=events[run])

    # set up contrasts: one per condition
    conditions = events[run].trial_type.unique()
    for condition_ in conditions:
        z_maps.append(glm.compute_contrast(condition_))
        conditions_label.append(condition_)
        run_label.append(run)

# %%
# Generating a report
# -------------------
# Since we have already computed the FirstLevelModel
# and have the :term:`contrast`, we can quickly create a summary report.

from nilearn.image import mean_img

mean_img_ = mean_img(func_filename)
report = glm.generate_report(
    contrasts=conditions,
    bg_img=mean_img_,
)

# %%
# This report can be viewed in a notebook.
report

# %%
# In a jupyter notebook, the report will be automatically inserted, as above.

# We can access the report via a browser:
# report.open_in_browser()

# Or we can save as an html file.
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_haxby_glm_decoding"
output_dir.mkdir(exist_ok=True, parents=True)
report.save_as_html(output_dir / "report.html")

# %%
# Build the decoding pipeline
# ---------------------------
# To define the decoding pipeline we use Decoder object, we choose :
#
# * a prediction model, here a Support Vector Classifier,
#   with a linear kernel
#
# * the mask to use, here a ventral temporal ROI in the visual cortex
#
# * although it usually helps to decode better, z-maps time series don't
#   need to be rescaled to a 0 mean, variance of 1 so we use
#   standardize=False.
#
# * we use univariate feature selection to reduce the dimension of the
#   problem keeping only 5% of voxels which are most informative.
#
# * a cross-validation scheme, here we use LeaveOneGroupOut
#   cross-validation on the runs which corresponds
#   to a leave-one-run-out
#
# We fit directly this pipeline on the Niimgs outputs of the GLM, with
# corresponding conditions labels and run labels
# (for the cross validation).

from sklearn.model_selection import LeaveOneGroupOut

from nilearn.decoding import Decoder

decoder = Decoder(
    estimator="svc",
    mask=haxby_dataset.mask,
    standardize=False,
    screening_percentile=5,
    cv=LeaveOneGroupOut(),
    verbose=1,
)
decoder.fit(z_maps, conditions_label, groups=run_label)

# Return the corresponding mean prediction accuracy compared to chance
# for classifying one-vs-all items.

classification_accuracy = np.mean(list(decoder.cv_scores_.values()))
chance_level = 1.0 / len(np.unique(conditions))
print(
    f"Classification accuracy: {classification_accuracy:.4f} / "
    f"Chance level: {chance_level}"
)
