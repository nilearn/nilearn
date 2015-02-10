"""
Basic Atlas interaction
=======================

Plot the regions of a reference atlas.
"""

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

yeo_2011_atlas_dataset = datasets.fetch_yeo_2011_atlas()
print yeo_2011_atlas_dataset.description

plotting.plot_roi(yeo_2011_atlas_dataset.thick_17,
                  title="yeo 2011 atlas")
plt.show()
