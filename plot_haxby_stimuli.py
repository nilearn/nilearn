"""Plots the Haxby stimuli"""
from scipy.misc import imread


def plot_stimuli(stimulus_information=None):

    if stimulus_information is None:
        from nilearn.datasets import fetch_haxby
        stimulus_information = fetch_haxby(n_subjects=0,
                                           fetch_stimuli=True).stimuli
    import pylab as plt
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


if __name__ == "__main__":
    plot_stimuli()

