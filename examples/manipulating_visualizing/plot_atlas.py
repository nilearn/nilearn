"""
Basic Atlas interaction
=======================

Plot the regions of a reference atlas.
"""

import matplotlib.pyplot as plt
from nilearn.datasets import fetch_yeo_2011_atlas
from nilearn import plotting

atlas = fetch_yeo_2011_atlas()
print atlas.description

plotting.plot_roi(atlas.thick_17, title="yeo 2011 atlas")
plt.show()
