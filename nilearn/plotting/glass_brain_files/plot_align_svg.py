"""The goal of this script is to align the glass brain SVGs on top of the
anatomy. This is only useful for internal purposes especially when the
SVG is modified.
"""

import functools

import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.plotting import img_plotting, glass_brain

plt.close('all')


# plotting anat for coarse alignment
bg_img, _, _, _ = img_plotting._load_anat()
display = img_plotting.plot_glass_brain(bg_img, threshold=0, black_bg=True,
                                        title='anat', alpha=1)
display = img_plotting.plot_glass_brain(bg_img, threshold=0,
                                        title='anat', alpha=1)


# plotting slices for finer alignment
# e.g. parieto-occipital sulcus

def add_brain_schematics(display):
    for axes in display.axes.itervalues():
        kwargs = {'alpha': 0.5, 'linewidth': 1, 'edgecolor': 'orange'}
        object_bounds = glass_brain.plot_brain_schematics(axes.ax,
                                                          axes.direction,
                                                          **kwargs)
        axes.add_object_bounds(object_bounds)


# side
display = plotting.plot_anat(display_mode='x', cut_coords=[-2])
add_brain_schematics(display)

# top
display = plotting.plot_anat(display_mode='z', cut_coords=[20])
add_brain_schematics(display)

# front
display = plotting.plot_anat(display_mode='y', cut_coords=[-20])
add_brain_schematics(display)

# all in one
display = plotting.plot_anat(display_mode='ortho', cut_coords=(-2, -20, 20))
add_brain_schematics(display)

# Plot multiple slices
display = plotting.plot_anat(display_mode='x')
add_brain_schematics(display)

display = plotting.plot_anat(display_mode='y')
add_brain_schematics(display)

display = plotting.plot_anat(display_mode='z')
add_brain_schematics(display)

plt.show()
