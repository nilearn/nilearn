"""
Massively univariate analysis of a motor task from the Localizer dataset
========================================================================

This example shows the results obtained in a massively univariate
analysis performed at the inter-subject level. 

"""
# Author: Bertrand Thirion, <bertrand.thirion@inria.fr>, Dec. 2015
print __doc__

import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

### Load Localizer contrast ###################################################
n_subjects = 20
localizer_dataset = datasets.fetch_localizer_contrasts(
    ['left vs right button press'], get_anats=True, n_subjects=n_subjects)
# print basic information on the dataset
print('First contrast nifti image (3D) is located at: %s' %
      localizer_dataset.cmaps[0])
contrast_map_filenames = localizer_dataset.cmaps
anat = localizer_dataset.anats[0]  # for display

### Mask data #################################################################
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(contrast_map_filenames)

### One-sample t-test ###############################################
from scipy.stats import ttest_1samp
_, pvals = ttest_1samp(fmri_masked, 0)
neg_log_pvals = -np.log10(pvals)
neg_log_pvals_unmasked = nifti_masker.inverse_transform(neg_log_pvals)

### Visualization #############################################################
from nilearn.plotting import plot_stat_map, show
threshold = - np.log10(0.01)  # 1% uncorrected

# Plot p-values
fig = plt.figure(figsize=(5, 3), facecolor='k')
display = plot_stat_map(neg_log_pvals_unmasked,
                        threshold=threshold, bg_img=anat,
                        display_mode='x', cut_coords=1,
                        figure=fig, black_bg=True)
title = ('Negative $\log_{10}$ p-values')
display.title(title, y=1.1)

# This is too dark ? A brighter version
fig = plt.figure(figsize=(5, 3), facecolor='k')
display = plot_stat_map(neg_log_pvals_unmasked,
                        threshold=threshold, bg_img=anat,
                        display_mode='x', cut_coords=1,
                        figure=fig, black_bg=True, dim=0)
title = ('Negative $\log_{10}$ p-values')
display.title(title, y=1.1)

show()
