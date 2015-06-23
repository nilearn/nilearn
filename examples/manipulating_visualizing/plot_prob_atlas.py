"""
Plotting 4D Atlas maps or regions onto the anatomical image
===========================================================

We show here how to plot the atlas maps or statistical maps which are
4D using a three different display types, for example using
1. "contours", ROIs contours are delineated by colored lines
2. "filled_contours", plots can be seen in contours with fillings
3. "continuous or overlays", plots can be seen in continuous overlays

This function can display each map with each different color which are picked
randomly from the given colormap.
Please see the related documentation for more information.
"""
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn import plotting

# Fetch the 4D atlas data or 4D statistical output data
# Harvard Oxford Atlas
atlas_filename, labels = datasets.fetch_harvard_oxford('cort-prob-2mm')

# Plotting Harvard Oxford Atlas maps as a contours type
# using view_type='contours'
display = plotting.img_plotting.plot_prob_atlas(atlas_filename,
                                                view_type='contours',
                                                title='Harvard-Oxford atlas '
                                                      'maps as contours')
# Plotting Harvard Oxford Atlas maps as a contours filled regions
# using view_type='filled_contours'
display = plotting.img_plotting.plot_prob_atlas(atlas_filename,
                                                view_type='filled_contours',
                                                title='Harvard-Oxford atlas '
                                                      'maps as a '
                                                      'contours and contour fillings')
# Plotting Harvard Oxford Atlas maps as a continuous type
# using view_type='continuous'
display = plotting.img_plotting.plot_prob_atlas(atlas_filename,
                                                view_type='continuous',
                                                title='Harvard-Oxford atlas '
                                                      'maps as a '
                                                      'continuous overlays')
plt.show()
