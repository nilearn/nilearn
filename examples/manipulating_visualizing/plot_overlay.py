"""
Visualizing a probablistic atlas: the default mode in the MSDL atlas
=====================================================================

Visualizing a probablistic atlas requires visualizing the different
maps that compose it.

Here we represent the nodes constituting the default mode network in the
`MSDL atlas
<https://team.inria.fr/parietal/research/spatial_patterns/spatial-patterns-in-resting-state/>`_.

The tools that we need to leverage are:

 * :func:`nilearn.image.index_img` to retrieve the various maps composing
   the atlas

 * Adding overlays on an existing brain display, to plot each of these
   maps

"""

import matplotlib.pyplot as plt
from nilearn import datasets, plotting, image

atlas_data = datasets.fetch_msdl_atlas()
atlas_filename = atlas_data.maps

# First plot the map for the PCC: index 4 in the atlas
display = plotting.plot_stat_map(image.index_img(atlas_filename, 4),
                                 colorbar=False,
                                 title="DMN nodes in MSDL atlas")

# Now add as an overlay the maps for the ACC and the left and right
# parietal nodes
display.add_overlay(image.index_img(atlas_filename, 5),
                    cmap=plotting.cm.black_blue)
display.add_overlay(image.index_img(atlas_filename, 6),
                    cmap=plotting.cm.black_green)
display.add_overlay(image.index_img(atlas_filename, 3),
                    cmap=plotting.cm.black_pink)

plt.show()
