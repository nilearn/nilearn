""" 
Perform a one-sample t-test on a bunch of images (a.k.a. second-level analyis in fMRI) and threshold a statistical image.

Author: Bertrand.thirion, Virgile Fritsch, 2014--2015
"""
print(__doc__)

import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

# Load some contrast images
n_samples = 20
localizer_dataset = datasets.fetch_localizer_calculation_task(
    n_subjects=n_samples)

# mask data
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
cmap_filenames = localizer_dataset.cmaps
fmri_masked = nifti_masker.fit_transform(cmap_filenames)

# perform a one-sample test on these values
from scipy.stats import ttest_1samp
_, p_values = ttest_1samp(fmri_masked, 0)

# z-transform of p-values
from nistats.utils import z_score
z_map = nifti_masker.inverse_transform(z_score(p_values))

# Threshold the resulting map:
# false positive rate < .001, cluster size > 10 voxels
from nistats.thresholding import map_threshold
thresholded_map1 = map_threshold(
    z_map, nifti_masker.mask_img_, threshold=.001, height_control='fpr',
    cluster_threshold=10)

# Now use FDR <.05, no lcuster-level threshold
thresholded_map2 = map_threshold(
    z_map, nifti_masker.mask_img_, threshold=.01, height_control='fdr')


# Visualization
from nilearn.plotting import plot_stat_map
display = plot_stat_map(z_map, title='Raw z map')
plot_stat_map(thresholded_map1, cut_coords=display.cut_coords,
              title='Thresholded z map, fpr <.001, clusters > 10 voxels')
plot_stat_map(thresholded_map2, cut_coords=display.cut_coords,
              title='Thresholded z map, expected fdr = .01')

plt.show()
