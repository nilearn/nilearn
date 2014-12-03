"""The goal of this script is to align the glass brain SVGs on top of the
anatomy. This is only useful for internal purposes especially when the
SVG is modified.
"""

import functools

import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.plotting import glass_brain

plt.close('all')


def add_brain_schematics(display, **kwargs):
    for axes in display.axes.itervalues():
        object_bounds = glass_brain.plot_brain_schematics(axes.ax,
                                                          axes.direction,
                                                          **kwargs)
        axes.add_object_bounds(object_bounds)

my_add_brain_schematics = functools.partial(add_brain_schematics,
                                            alpha=0.5,
                                            linewidth=1,
                                            edgecolor='orange')
# side
display = plotting.plot_anat(display_mode='x', cut_coords=[-2])
my_add_brain_schematics(display)

# top
display = plotting.plot_anat(display_mode='z', cut_coords=[20])
my_add_brain_schematics(display)

# front
display = plotting.plot_anat(display_mode='y', cut_coords=[-20])
my_add_brain_schematics(display)

# all in one
display = plotting.plot_anat(display_mode='ortho', cut_coords=(-2, -20, 20))
my_add_brain_schematics(display)

# Plot multiple slices
display = plotting.plot_anat(display_mode='x')
my_add_brain_schematics(display)

display = plotting.plot_anat(display_mode='y')
my_add_brain_schematics(display)

display = plotting.plot_anat(display_mode='z')
my_add_brain_schematics(display)

plt.show()
