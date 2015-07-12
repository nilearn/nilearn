"""
Visualizing 4D atlas maps onto the anatomical image
===========================================================

Basically this example gives an idea on how to visualize the atlas maps which are
4D using three different display types, for instance if you choose

1. "contours", which means maps or ROIs are shown as contours delineated by \
    colored lines.

2. "filled_contours", maps are shown as contours same as above but with \
    fillings inside the contours.

3. "continuous", maps are shown as just color overlays.

This function can display each map with each different color which are picked
randomly from the colormap which is already defined.

Please see the related documentation for more information to tune between the
parameters.
"""
# Load 4D Atlas maps
print('--- Loading 4D Atlas Maps---')
from nilearn import datasets

#### Harvard Oxford Atlas ####
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
#### Multi Subject Dictionary Learning Atlas ####
msdl = datasets.fetch_atlas_msdl()
#### Smith ICA Atlas and Brain Maps 2009 ####
smith = datasets.fetch_atlas_smith_2009()
#### Craddock Parcellations 2012 ####
craddock = datasets.fetch_atlas_craddock_2012()

# Visualization
print('--- Visualizing ---')
import matplotlib.pyplot as plt
from nilearn import plotting

display_types = ['contours', 'filled_contours', 'continuous']
atlas_types = [harvard_oxford.maps, msdl.maps, smith.rsn10, craddock.scorr_2level]
atlas_names = ['Harvard_Oxford', 'MSDL', 'Smith2009', 'Craddock2012']
for atlas, name in zip(atlas_types, atlas_names):
    for option in display_types:
        plotting.img_plotting.plot_prob_atlas(atlas, view_type=option,
                                              title='%s as %s' % (name, option))

plt.show()

