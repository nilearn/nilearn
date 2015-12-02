"""
Plotting tips
=============

This example shows how to display an activation image on top of an anatomical
one. 
"""
# Author: Bertrand Thirion, <bertrand.thirion@inria.fr>, Dec. 2015
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
from nilearn.input_data import NiftiMasker


localizer_dataset = datasets.fetch_localizer_contrasts(
    ["left vs right button press"],
    n_subjects=2,
    get_anats=True,
    get_tmaps=True)
localizer_anat_filename = localizer_dataset.anats[1]
localizer_tmap_filename = localizer_dataset.tmaps[1]

###############################################################################
# Plotting a map on MNI template
plotting.plot_stat_map(localizer_tmap_filename, threshold=3,
                       title="Plot on MNI152 template", display_mode='x',
                       cut_coords=(36, -27, 66))

###############################################################################
# Plotting a map with bakground anatomy
plotting.plot_stat_map(localizer_tmap_filename, bg_img=localizer_anat_filename,
                       threshold=3, title="Plot on subject's anatomy",
                       display_mode='x', cut_coords=(36, -27, 66))

###############################################################################
# Plotting on a brighter anatomy
plotting.plot_stat_map(localizer_tmap_filename, bg_img=localizer_anat_filename,
                       threshold=3, title="Plot on brighter anatomical image",
                       display_mode='x', cut_coords=(36, -27, 66), dim=0)


plotting.show()
