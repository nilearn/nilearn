"""Generation and plot of a events.tsv file generation: the neurospin/localizer events.
=======================================================================================

The protocol described is the so-called "archi standard" localizer
event sequence.  See Pinel et al., BMC neuroscience 2007 for reference.
"""

print(__doc__)

#########################################################################
# Define the onset times in seconds. Those are typically extracted
# from the stimulation software used.
import numpy as np
onset = np.array([
    0., 2.4, 8.7, 11.4, 15., 18., 20.7, 23.7, 26.7, 29.7, 33., 35.4, 39.,
    41.7, 44.7, 48., 56.4, 59.7, 62.4, 69., 71.4, 75., 83.4, 87., 89.7,
    96., 108., 116.7, 119.4, 122.7, 125.4, 131.4, 135., 137.7, 140.4,
    143.4, 146.7, 149.4, 153., 156., 159., 162., 164.4, 167.7, 170.4,
    173.7, 176.7, 188.4, 191.7, 195., 198., 201., 203.7, 207., 210.,
    212.7, 215.7, 218.7, 221.4, 224.7, 227.7, 230.7, 234., 236.7, 246.,
    248.4, 251.7, 254.7, 257.4, 260.4, 264., 266.7, 269.7, 275.4, 278.4,
    284.4, 288., 291., 293.4, 296.7])

#########################################################################
# Associated trial types: these are numbered between 0 and 5, hence
# correspond to 6 different conditions.
trial_idx = np.array(
    [3, 3, 0, 2, 5, 3, 5, 2, 3, 5, 1, 2, 4, 4, 2, 2, 4, 0, 2, 3, 3, 4, 2,
     2, 5, 1, 2, 3, 5, 1, 3, 4, 2, 2, 1, 2, 5, 0, 3, 1, 4, 2, 3, 4, 2, 2,
     0, 0, 2, 4, 3, 3, 1, 1, 1, 3, 3, 0, 3, 0, 3, 2, 3, 5, 4, 0, 2, 2, 2,
     3, 1, 0, 0, 3, 1, 5, 4, 3, 5, 5])

#########################################################################
# We may want to map these indices to explicit condition names.
# For that, we define a list of 10 strings.
condition_ids = ['horizontal checkerboard',
                 'vertical checkerboard',
                 'auditory instructions',
                 'visual instructions',
                 'visual sentence',
                 'auditory sentence']

trial_type = np.array([condition_ids[i] for i in trial_idx])

#########################################################################
# We also define a duration (required by BIDS conventions).
duration = np.ones_like(onset)


#########################################################################
# Form an event dataframe from these information.
import pandas as pd
events = pd.DataFrame({'trial_type': trial_type,
                       'onset': onset,
                       'duration': duration})

#########################################################################
# Export them to a tsv file.
tsvfile = 'localizer_events.tsv'
events.to_csv(tsvfile, sep='\t', index=False)
print("Created the events file in %s " % tsvfile)


#########################################################################
# Plot the event dataframe.
import matplotlib.pyplot as plt
from nilearn.reporting import plot_event
plot_event(events, figsize=(12, 4))
plt.show()