"""
Show stimuli of Haxby dataset
=============================

.. admonition:: dataset

    This example plots an overview of the stimuli used
    in the :ref:`Haxby dataset <haxby_dataset>`.

"""

# %%
from nilearn import datasets
from nilearn.plotting import show

haxby_dataset = datasets.fetch_haxby(n_subjects=1, fetch_stimuli=True)
stimulus_information = haxby_dataset.stimuli

# %%
import matplotlib.pyplot as plt

for stim_type in stimulus_information:
    # skip control images, there are too many
    if stim_type != "controls":
        file_names = stimulus_information[stim_type]

        fig, axes = plt.subplots(6, 8)
        fig.suptitle(stim_type)

        for img_path, ax in zip(file_names, axes.ravel(), strict=False):
            ax.imshow(plt.imread(img_path), cmap="gray")

        for ax in axes.ravel():
            ax.axis("off")

show()

# sphinx_gallery_dummy_images=7
