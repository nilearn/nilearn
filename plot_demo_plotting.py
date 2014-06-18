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
                                               get_anats=True)


###############################################################################
# demo the different plotting function

# Plotting statistical maps
plotting.plot_stat_map(localizer.cmaps[3], bg_img=localizer.anats[3],
                       title="plot_stat_map")

# Plotting anatomical maps
plotting.plot_anat(haxby.anat[0], title="plot_anat")

# Plotting ROIs (here the mask)
plotting.plot_roi(haxby.mask_vt[0], bg_img=haxby.anat[0], title="plot_roi")

# Plotting EPI haxby
plotting.plot_epi(image.mean_img(haxby.func[0]), title="plot_epi")

###############################################################################
# demo the different display_mode

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
                       title="display_mode='yz', cut_coords=(-27, 36)")

plotting.plot_stat_map(localizer.cmaps[3], display_mode='yz',
                       cut_coords=(-27, 60),
                       title="display_mode='yz', cut_coords=(-27, 60)")

import pylab as pl
pl.show()
