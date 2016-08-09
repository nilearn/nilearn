"""
Show stimuli of Haxby et al. dataset
===============================================================================

In this script we plot an overview of the stimuli used in "Distributed
and Overlapping Representations of Faces and Objects in Ventral Temporal
Cortex" (Science 2001)
"""

from scipy.misc import imread
import matplotlib.pyplot as plt

from nilearn import datasets

haxby_dataset = datasets.fetch_haxby(subjects=[], fetch_stimuli=True)
stimulus_information = haxby_dataset.stimuli

for stim_type in sorted(stimulus_information.keys()):
    if stim_type == b'controls':
        # skip control images, there are too many
        continue

    file_names = stimulus_information[stim_type]

    plt.figure()
    for i in range(48):
        plt.subplot(6, 8, i + 1)
        try:
            plt.imshow(imread(file_names[i]), cmap=plt.cm.gray)
        except:
            # just go to the next one if the file is not present
            pass
        plt.axis("off")
    plt.suptitle(stim_type)

plt.show()
