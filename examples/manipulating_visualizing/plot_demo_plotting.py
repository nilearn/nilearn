"""
Demoing Plotting functions of nilearn
======================================

Nilearn comes with a set of plotting function for Nifti-like images.

"""

from nilearn import datasets
from nilearn import plotting, image


###############################################################################
# Retrieve the data: haxby dataset to have EPI images and masks, and
# localizer dataset to have contrast maps

haxby_dataset = datasets.fetch_haxby(n_subjects=1)
haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=4,
    get_anats=True,
    get_tmaps=True)
localizer_anat_filename = localizer_dataset.anats[3]
localizer_cmap_filename = localizer_dataset.cmaps[3]
localizer_tmap_filename = localizer_dataset.tmaps[3]

###############################################################################
# demo the different plotting function

# Plotting statistical maps
plotting.plot_stat_map(localizer_cmap_filename, bg_img=localizer_anat_filename,
                       threshold=3, title="plot_stat_map",
                       cut_coords=(36, -27, 66))

# Plotting anatomical maps
plotting.plot_anat(haxby_anat_filename, title="plot_anat")

# Plotting ROIs (here the mask)
plotting.plot_roi(haxby_mask_filename, bg_img=haxby_anat_filename,
                  title="plot_roi")

# Plotting EPI haxby
mean_haxby_img = image.mean_img(haxby_func_filename)
plotting.plot_epi(mean_haxby_img, title="plot_epi")

# Plotting glass brain
plotting.plot_glass_brain(localizer_tmap_filename, title='plot_glass_brain',
                          threshold=3)

plotting.plot_glass_brain(localizer_tmap_filename, title='plot_glass_brain',
                          black_bg=True, display_mode='xz', threshold=3)

###############################################################################
# demo the different display_mode

plotting.plot_stat_map(localizer_cmap_filename, display_mode='ortho',
                       cut_coords=(36, -27, 60),
                       title="display_mode='ortho', cut_coords=(36, -27, 60)")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='z', cut_coords=5,
                       title="display_mode='z', cut_coords=5")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='x', cut_coords=(-36, 36),
                       title="display_mode='x', cut_coords=(-36, 36)")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='y', cut_coords=1,
                       title="display_mode='x', cut_coords=(-36, 36)")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='z',
                       cut_coords=1, colorbar=False,
                       title="display_mode='z', cut_coords=1, colorbar=False")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='xz',
                       cut_coords=(36, 60),
                       title="display_mode='xz', cut_coords=(36, 60)")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='yx',
                       cut_coords=(-27, 36),
                       title="display_mode='yx', cut_coords=(-27, 36)")

plotting.plot_stat_map(localizer_cmap_filename, display_mode='yz',
                       cut_coords=(-27, 60),
                       title="display_mode='yz', cut_coords=(-27, 60)")

###############################################################################
# demo the outline modes

# Plot T1 outline on top of the mean EPI (useful for checking coregistration)
my_plot = plotting.plot_anat(mean_haxby_img, title="add_edges")
my_plot.add_edges(haxby_anat_filename)

# Plotting outline of the mask on top of the EPI
my_plot = plotting.plot_anat(mean_haxby_img, title="add_contours",
                             cut_coords=(28, -34, -22))
my_plot.add_contours(haxby_mask_filename, levels=[0.5], colors='r')

import pylab as pl
pl.show()
