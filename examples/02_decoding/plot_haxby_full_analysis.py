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

.. include:: ../../../examples/masker_note.rst

"""

##########################################################################
# Load and prepare the data
# -----------------------------------

# Fetch data using nilearn dataset fetcher
from nilearn import datasets

# by default we fetch 2nd subject data for analysis
haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]

# Print basic information on the dataset
print(
    "First subject anatomical nifti image (3D) located is "
    f"at: {haxby_dataset.anat[0]}"
)
print(
    "First subject functional nifti image (4D) is located "
    f"at: {func_filename}"
)

# load labels
import pandas as pd

# Load nilearn NiftiMasker, the practical masking and unmasking tool
from nilearn.maskers import NiftiMasker

labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
stimuli = labels["labels"]
# identify resting state labels in order to be able to remove them
task_mask = stimuli != "rest"

# find names of remaining active labels
categories = stimuli[task_mask].unique()

# extract tags indicating to which acquisition run a tag belongs
session_labels = labels["chunks"][task_mask]

# apply the task_mask to  fMRI data (func_filename)
from nilearn.image import index_img

task_data = index_img(func_filename, task_mask)

##########################################################################
# Decoding on the different masks
# --------------------------------
#
# The classifier used here is a support vector classifier (svc). We use
# class:`nilearn.decoding.Decoder` and specify the classifier.
import numpy as np
from nilearn.decoding import Decoder

# Make a data splitting object for cross validation
from sklearn.model_selection import LeaveOneGroupOut

cv = LeaveOneGroupOut()

##############################################################
# We use :class:`nilearn.decoding.Decoder` to estimate a baseline.

mask_names = ["mask_vt", "mask_face", "mask_house"]

mask_scores = {}
mask_chance_scores = {}

for mask_name in mask_names:
    print(f"Working on {mask_name}")
    # For decoding, standardizing is often very important
    mask_filename = haxby_dataset[mask_name][0]
    masker = NiftiMasker(mask_img=mask_filename, standardize=True)
    mask_scores[mask_name] = {}
    mask_chance_scores[mask_name] = {}

    for category in categories:
        print(f"Processing {mask_name} {category}")
        classification_target = stimuli[task_mask] == category
        # Specify the classifier to the decoder object.
        # With the decoder we can input the masker directly.
        # We are using the svc_l1 here because it is intra subject.
        decoder = Decoder(
            estimator="svc_l1", cv=cv, mask=masker, scoring="roc_auc"
        )
        decoder.fit(task_data, classification_target, groups=session_labels)
        mask_scores[mask_name][category] = decoder.cv_scores_[1]
        mean = np.mean(mask_scores[mask_name][category])
        std = np.std(mask_scores[mask_name][category])
        print(f"Scores: {mean:1.2f} +- {std:1.2f}")

        dummy_classifier = Decoder(
            estimator="dummy_classifier", cv=cv, mask=masker, scoring="roc_auc"
        )
        dummy_classifier.fit(
            task_data, classification_target, groups=session_labels
        )
        mask_chance_scores[mask_name][category] = dummy_classifier.cv_scores_[
            1
        ]


##########################################################################
# We make a simple bar plot to summarize the results
# ---------------------------------------------------
import matplotlib.pyplot as plt
from nilearn.plotting import show

plt.figure()

tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, mask_name in zip("rgb", mask_names):
    score_means = [
        np.mean(mask_scores[mask_name][category]) for category in categories
    ]
    plt.bar(
        tick_position, score_means, label=mask_name, width=0.25, color=color
    )

    score_chance = [
        np.mean(mask_chance_scores[mask_name][category])
        for category in categories
    ]
    plt.bar(
        tick_position,
        score_chance,
        width=0.25,
        edgecolor="k",
        facecolor="none",
    )

    tick_position = tick_position + 0.2

plt.ylabel("Classification accuracy (AUC score)")
plt.xlabel("Visual stimuli category")
plt.ylim(0.3, 1)
plt.legend(loc="lower right")
plt.title("Category-specific classification accuracy for different masks")
plt.tight_layout()


show()
