import matplotlib.pyplot as plt
import matplotlib.transforms

from nilearn import plotting
from nilearn.plotting import slicers

plt.close('all')


# side
display = plotting.plot_anat(display_mode='x', cut_coords=[-2])
display.add_brain_schematics()

# top
display = plotting.plot_anat(display_mode='z', cut_coords=[20])
display.add_brain_schematics()

# front
display = plotting.plot_anat(display_mode='y', cut_coords=[-20])
display.add_brain_schematics()

# all in one
display = plotting.plot_anat(display_mode='ortho', cut_coords=(-2, -20, 20))
display.add_brain_schematics()

# Plot multiple slices
display = plotting.plot_anat(display_mode='x')
display.add_brain_schematics()

display = plotting.plot_anat(display_mode='y')
display.add_brain_schematics()

display = plotting.plot_anat(display_mode='z')
display.add_brain_schematics()

plt.show()

# Leaving that here for legacy params
# # all in one
# display = plotting.plot_anat(display_mode='ortho', cut_coords=(-2, -20, 20))
# axes = [each.ax for each in display.axes.itervalues()]

# ax = axes[0]
# bp = brain_plotter.BrainPlotter('generated_json/brain_schematics_front.json')
# transform = matplotlib.transforms.Affine2D.from_values(0.38, 0, 0, 0.38, -71, -73)
# transform += ax.transData
# bp.plot(ax, transform)

# ax = axes[1]
# bp = brain_plotter.BrainPlotter('generated_json/brain_schematics_side.json')
# transform = matplotlib.transforms.Affine2D.from_values(0.36, 0, 0, 0.36, -103, -67)
# transform += ax.transData
# bp.plot(ax, transform)

# ax = axes[2]
# bp = brain_plotter.BrainPlotter('generated_json/brain_schematics_top.json')
# transform = matplotlib.transforms.Affine2D.from_values(0.35, 0, 0, 0.35, -70, -103)
# transform += ax.transData
# bp.plot(ax, transform)
