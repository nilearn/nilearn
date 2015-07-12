"""
Plotting in nilearn
==========================

Nilearn comes with a set of plotting function for Nifti-like images,
see :ref:`plotting` for more details.
"""

from nilearn import datasets
from nilearn import plotting, image

###############################################################################
# Retrieve the data: haxby dataset to have EPI images and masks, and
# localizer dataset to have contrast maps

haxby_dataset = datasets.fetch_haxby(n_subjects=1)

# print basic information on the dataset
print('First subject anatomical nifti image (3D) is at: %s' %
      haxby_dataset.anat[0])
print('First subject functional nifti image (4D) is at: %s' %
      haxby_dataset.func[0])  # 4D data

haxby_anat_filename = haxby_dataset.anat[0]
haxby_mask_filename = haxby_dataset.mask_vt[0]
haxby_func_filename = haxby_dataset.func[0]

localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=2,
    get_anats=True,
    get_tmaps=True)
localizer_anat_filename = localizer_dataset.anats[1]
localizer_tmap_filename = localizer_dataset.tmaps[1]

###############################################################################
# demo the different plotting functions

# Plotting statistical maps
plotting.plot_stat_map(localizer_tmap_filename, bg_img=localizer_anat_filename,
                       threshold=3, title="plot_stat_map",
                       cut_coords=(36, -27, 66))

# Plotting glass brain
plotting.plot_glass_brain(localizer_tmap_filename, title='plot_glass_brain',
                          threshold=3)

# Plotting anatomical maps
plotting.plot_anat(haxby_anat_filename, title="plot_anat")

# Plotting ROIs (here the mask)
plotting.plot_roi(haxby_mask_filename, bg_img=haxby_anat_filename,
                  title="plot_roi")

# Plotting EPI haxby
mean_haxby_img = image.mean_img(haxby_func_filename)
plotting.plot_epi(mean_haxby_img, title="plot_epi")

import matplotlib.pyplot as plt
plt.show()
