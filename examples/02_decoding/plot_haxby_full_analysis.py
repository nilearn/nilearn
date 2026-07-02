"""
ROI-based decoding analysis in Haxby et al. dataset
===================================================

In this script we reproduce the data analysis
conducted by :footcite:t:`Haxby2001`.

Specifically, we look at decoding accuracy for different objects in
three different masks:

- the full ventral stream (mask_vt),
- the house selective areas (mask_house)
- and the face selective areas (mask_face).

The masks were defined via a standard GLM-based analysis.

"""

import warnings

warnings.filterwarnings(
    "ignore", message="The provided image has no sform in its header."
)
warnings.filterwarnings(
    "ignore", message="The decoding model will be trained only on"
)

# %%
# Load and prepare the data
# -------------------------
# We fetch the data a single subject for analysis.
# We also find the names of the different categories of stimuli,
# identify in which run they were presented.

import pandas as pd

from nilearn import datasets

haxby_dataset = datasets.fetch_haxby()
func_filename = haxby_dataset.func[0]

labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")

stimuli = labels["labels"]

task_mask = stimuli != "rest"

categories = stimuli[task_mask].unique()
run_labels = labels["chunks"][task_mask]

# %%
# We index the volumes that do NOT correspond to a rest condition.
from nilearn.image import index_img

task_data = index_img(func_filename, task_mask)

# %%
# Decoding on the different masks
# -------------------------------
#
# The classifier used here is a support vector classifier (svc).
# We use
# :class:`~nilearn.decoding.Decoder` and specify the classifier.
# We are using the ``svc_l1`` here because it is intra subject.
# We will be doing a 'leave one run out' for cross validation.
#
# We use Nilearn's NiftiMasker to extract and standardize
# the data from the nifti timeseries:
# the masker can directly be passed to the decoder object.
#
# We will use :class:`~nilearn.decoding.Decoder`
# with a "dummy_classifier" to estimate a baseline.
#

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from nilearn.decoding import Decoder
from nilearn.maskers import NiftiMasker

cv = LeaveOneGroupOut()

mask_names = ["mask_vt", "mask_face", "mask_house"]

mask_scores = {}
mask_chance_scores = {}

for mask_name in mask_names:
    print(f"Working on {mask_name}")

    mask_filename = haxby_dataset[mask_name][0]
    masker = NiftiMasker(mask_img=mask_filename, standardize="zscore_sample")

    mask_scores[mask_name] = {}
    mask_chance_scores[mask_name] = {}

    for category in categories:
        print(f" Processing {category}")

        classification_target = stimuli[task_mask] == category

        decoder = Decoder(
            estimator="svc_l1",
            cv=cv,
            mask=masker,
            scoring="roc_auc",
            screening_percentile=100,
            standardize="zscore_sample",
        )
        decoder.fit(task_data, classification_target, groups=run_labels)

        mask_scores[mask_name][category] = decoder.cv_scores_[1]

        mean = np.mean(mask_scores[mask_name][category])
        std = np.std(mask_scores[mask_name][category])
        print(f"  Scores: {mean:1.2f} +- {std:1.2f}")

        dummy_classifier = Decoder(
            estimator="dummy_classifier",
            cv=cv,
            mask=masker,
            scoring="roc_auc",
            screening_percentile=100,
            standardize="zscore_sample",
        )
        dummy_classifier.fit(
            task_data, classification_target, groups=run_labels
        )

        mask_chance_scores[mask_name][category] = dummy_classifier.cv_scores_[
            1
        ]

# %%
# We make a simple bar plot to summarize the results
# --------------------------------------------------
import matplotlib.pyplot as plt

from nilearn.plotting import show

plt.figure(constrained_layout=True)

tick_position = np.arange(len(categories))
plt.xticks(tick_position, categories, rotation=45)

for color, mask_name in zip("rgb", mask_names, strict=False):
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
plt.legend(loc="upper right")
plt.title("Category-specific classification accuracy for different masks")

show()

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=1
