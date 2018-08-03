"""
Nilearn Colormaps
=======================

Visualize all available color maps from Nilearn's plotting library.
They are plotted on top of a colormap test image, designed to judge a
colormap's perceptual uniformity, as shown here:
https://peterkovesi.com/projects/colourmaps/colourmaptestimage.html
"""
import os.path as op

import matplotlib.pyplot as plt

from nilearn.plotting.cm import _cmap_d

##########################################################################
# Load the image
# -------------------------

img = plt.imread(op.join(op.dirname(__file__)), 'colourmaptest.tif')


###########################################################################
# Plot image with all of Nilearn's color maps.
# ------------------------------------

nmaps = len(_cmap_d)
fig, axes = plt.subplots(nrows=nmaps // 2, ncols=2, dpi=200,
                         figsize=2 * plt.figaspect(8))

for cmap, ax in zip(_cmap_d.keys(), axes.flatten()):
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    ax.set_title(cmap, fontsize=8)

fig.subplots_adjust(wspace=0.05, hspace=0.)
plt.show()
