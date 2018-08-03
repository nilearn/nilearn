"""
Matplotlib colormaps in Nilearn
================================

Visualize HCP connectome workbench color maps shipped with Nilearn
which can be used for plotting brain images on surface.

See :ref:`surface-plotting` for surface plotting details.
"""
import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting.cm import _cmap_d as hcp_cmaps

###########################################################################
# Plot color maps
# ----------------

nmaps = len(hcp_cmaps)
a = np.outer(np.arange(0, 1, 0.01), np.ones(10))

# Initialize the figure
plt.figure(figsize=(10, 4.2))
plt.subplots_adjust(top=0.4, bottom=0.05, left=0.01, right=0.99)

for index, cmap in enumerate(hcp_cmaps):
    plt.subplot(1, nmaps + 1, index + 1)
    plt.imshow(a, cmap=hcp_cmaps[cmap])
    plt.axis('off')
    plt.title(cmap, fontsize=10, va='bottom', rotation=90)

plt.show()
