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

# Harvard Oxford Atlas
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
harvard_oxford_sub = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')

# Multi Subject Dictionary Learning Atlas
msdl = datasets.fetch_atlas_msdl()

# Smith ICA Atlas and Brain Maps 2009
smith = datasets.fetch_atlas_smith_2009()

# ICBM tissue probability
icbm = datasets.fetch_icbm152_2009()

# Visualization
print('--- Visualizing ---')
import matplotlib.pyplot as plt
from nilearn import plotting

atlas_types = [harvard_oxford.maps, harvard_oxford_sub.maps,
               msdl.maps, smith.rsn10,
               smith.rsn20, smith.rsn70, smith.bm10,
               smith.bm20, smith.bm70,
               (icbm['wm'], icbm['gm'], icbm['csf'])]
atlas_names = ['Harvard_Oxford', 'Harvard_Oxford sub', 'MSDL',
               'Smith2009 10 RSNs', 'Smith2009 20 RSNs',
               'Smith2009 70 RSNs', 'Smith2009 10 Brainmap',
               'Smith2009 20 Brainmap', 'Smith2009 70 Brainmap',
               'ICBM tissues']

for atlas, name in zip(atlas_types, atlas_names):
        plotting.plot_prob_atlas(atlas,
                                              title='%s' % name)

plt.show()

