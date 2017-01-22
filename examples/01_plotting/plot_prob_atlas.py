"""
Visualizing 4D probabilistic atlas maps
=======================================

This example shows how to visualize probabilistic atlases made of 4D images.
There are 3 different display types:

1. "contours", which means maps or ROIs are shown as contours delineated by \
    colored lines.

2. "filled_contours", maps are shown as contours same as above but with \
    fillings inside the contours.

3. "continuous", maps are shown as just color overlays.

A colorbar can optionally be added.

The :func:`nilearn.plotting.plot_prob_atlas` function displays each map
with each different color which are picked randomly from the colormap
which is already defined.

See :ref:`plotting` for more information to know how to tune the parameters.
"""
# Load 4D probabilistic atlases
from nilearn import datasets

# Harvard Oxford Atlasf
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
harvard_oxford_sub = datasets.fetch_atlas_harvard_oxford('sub-prob-2mm')

# Multi Subject Dictionary Learning Atlas
msdl = datasets.fetch_atlas_msdl()

# Smith ICA Atlas and Brain Maps 2009
smith = datasets.fetch_atlas_smith_2009()

# ICBM tissue probability
icbm = datasets.fetch_icbm152_2009()

# Allen RSN networks
allen = datasets.fetch_atlas_allen_2011()

# Visualization
from nilearn import plotting

atlas_types = {'Harvard_Oxford': harvard_oxford.maps,
               'Harvard_Oxford sub': harvard_oxford_sub.maps,
               'MSDL': msdl.maps, 'Smith 2009 10 RSNs': smith.rsn10,
               'Smith2009 20 RSNs': smith.rsn20,
               'Smith2009 70 RSNs': smith.rsn70,
               'Smith2009 20 Brainmap': smith.bm20,
               'Smith2009 70 Brainmap': smith.bm70,
               'ICBM tissues': (icbm['wm'], icbm['gm'], icbm['csf']),
               'Allen2011': allen.rsn28,
               }

for name, atlas in sorted(atlas_types.items()):
    plotting.plot_prob_atlas(atlas, title=name)

# An optional colorbar can be set
plotting.plot_prob_atlas(smith.bm10, title='Smith2009 10 Brainmap (with'
                                           ' colorbar)',
                         colorbar=True)
plotting.show()
