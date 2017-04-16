"""
Show stimuli of Haxby et al. dataset
===============================================================================

In this script we plot an overview of the stimuli used in
"Distributed and Overlapping Representations of Faces and
    Objects in Ventral Temporal Cortex"

(Science 2001)
"""

from scipy.misc import imread

from nilearn.datasets import fetch_haxby
stimulus_information = fetch_haxby(n_subjects=0,
                                   fetch_stimuli=True).stimuli
import matplotlib.pyplot as plt
for stim_type in stimulus_information.keys():
    if stim_type == "controls":
        # skip control images, there are too many
        continue

    file_names = stimulus_information[stim_type]

    plt.figure()
    for i in range(48):
        plt.subplot(6, 8, i + 1)
        try:
            plt.imshow(imread(file_names[i]))

            plt.gray()
            plt.axis("off")
        except:
            # just go to the next one if the file is not present
            continue
    plt.suptitle(stim_type)
plt.show()

