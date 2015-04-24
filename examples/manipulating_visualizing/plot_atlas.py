"""
Basic Atlas plotting
=======================

Plot the regions of a reference atlas (here the Harvard-Oxford atlas).
"""

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

atlas_filename, labels = datasets.fetch_harvard_oxford('cort-maxprob-thr25-2mm')

print('Atlas ROIs are located at: %s' % atlas_filename)

plotting.plot_roi(atlas_filename, title="Harvard Oxford atlas")
plt.show()
