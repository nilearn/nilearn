"""
Demoing Plotting functions of nilearn
======================================

Nilearn comes with a set of plotting function for Nifti-like images.

"""

from nilearn import plotting, datasets, image


###############################################################################
# Retrieve the data: haxby dataset to have EPI images and masks, and
# localizer dataset to have contrast maps

haxby = datasets.fetch_haxby(n_subjects=1)

localizer = datasets.fetch_localizer_contrasts(["left vs right button press"],
                                               n_subjects=4,
                                               get_anats=True,
                                               get_tmaps=True)


###############################################################################
# demo the different plotting function

# Plotting statistical maps
plotting.plot_stat_map(localizer.cmaps[3], bg_img=localizer.anats[3],
                       threshold=3, title="plot_stat_map",
                       cut_coords=(36, -27, 66))

# Plotting anatomical maps
plotting.plot_anat(haxby.anat[0], title="plot_anat")

# Plotting ROIs (here the mask)
plotting.plot_roi(haxby.mask_vt[0], bg_img=haxby.anat[0], title="plot_roi")

# Plotting EPI haxby
mean_haxby_img = image.mean_img(haxby.func[0])
plotting.plot_epi(mean_haxby_img, title="plot_epi")

# Plotting glass brain
plotting.plot_glass_brain(localizer.tmaps[3], title='plot_glass_brain',
                          threshold=3)

plotting.plot_glass_brain(localizer.tmaps[3], title='plot_glass_brain',
                          black_bg=True, display_mode='xz', threshold=3)

###############################################################################
# demo the different display_mode

plotting.plot_stat_map(localizer.cmaps[3], display_mode='ortho',
                       cut_coords=(36, -27, 60),
                       title="display_mode='ortho', cut_coords=(36, -27, 60)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='z', cut_coords=5,
                       title="display_mode='z', cut_coords=5")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='x', cut_coords=(-36, 36),
                       title="display_mode='x', cut_coords=(-36, 36)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='y', cut_coords=1,
                       title="display_mode='x', cut_coords=(-36, 36)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='z',
                       cut_coords=1, colorbar=False,
                       title="display_mode='z', cut_coords=1, colorbar=False")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='xz',
                       cut_coords=(36, 60),
                       title="display_mode='xz', cut_coords=(36, 60)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='yx',
                       cut_coords=(-27, 36),
                       title="display_mode='yx', cut_coords=(-27, 36)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='yz',
                       cut_coords=(-27, 60),
                       title="display_mode='yz', cut_coords=(-27, 60)")

###############################################################################
# demo the outline modes

# Plot T1 outline on top of the mean EPI (useful for checking coregistration)
my_plot = plotting.plot_anat(mean_haxby_img, title="add_edges")
my_plot.add_edges(haxby.anat[0])

# Plotting outline of the mask on top of the EPI
my_plot = plotting.plot_anat(mean_haxby_img, title="add_contours",
                             cut_coords=(28,-34,-22))
my_plot.add_contours(haxby.mask_vt[0], levels=[0.5], colors='r')

import pylab as pl
pl.show()
