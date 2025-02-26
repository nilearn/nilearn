"""
Show stimuli of Haxby et al. dataset
====================================

In this script we plot an overview of the stimuli used
in :footcite:t:`Haxby2001`.
"""

from nilearn._utils.helpers import check_matplotlib

check_matplotlib()

import matplotlib.pyplot as plt

# %%
from nilearn import datasets
from nilearn.plotting import show

haxby_dataset = datasets.fetch_haxby(subjects=[], fetch_stimuli=True)
stimulus_information = haxby_dataset.stimuli

# %%

for stim_type in stimulus_information:
    # skip control images, there are too many
    if stim_type != "controls":
        file_names = stimulus_information[stim_type]

        fig, axes = plt.subplots(6, 8)
        fig.suptitle(stim_type)

        for img_path, ax in zip(file_names, axes.ravel()):
            ax.imshow(plt.imread(img_path), cmap="gray")

        for ax in axes.ravel():
            ax.axis("off")

show()

# %%
# References
# ----------
#
# .. footbibliography::


# sphinx_gallery_dummy_images=7
