"""
Example of simple second level analysis
=======================================

Perform a one-sample t-test on a bunch of images
(a.k.a. second-level analyis in fMRI) and threshold a statistical image.
This is based on the so-called localizer dataset.
It shows activation related to a mental computation task,
as opposed to narrative sentence reading/listening.

"""
from nilearn import datasets
from nilearn.input_data import NiftiMasker

#########################################################################
# Prepare some images for a simple t test
# ----------------------------------------
# This is a simple manually performed second level analysis
n_samples = 20
localizer_dataset = datasets.fetch_localizer_calculation_task(
    n_subjects=n_samples)

# mask data
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
cmap_filenames = localizer_dataset.cmaps

#########################################################################
# Perform the second level analysis
# ----------------------------------
# perform a one-sample test on these values
import pandas as pd
design_matrix = pd.DataFrame([1] * n_samples, columns=['intercept'])

from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel().fit(
    cmap_filenames, design_matrix=design_matrix)
z_map = second_level_model.compute_contrast(output_type='z_score')

#########################################################################
# Threshold the resulting map:
# false positive rate < .001, cluster size > 10 voxels
from nistats.thresholding import map_threshold
thresholded_map1, threshold1 = map_threshold(
    z_map, threshold=.001, height_control='fpr', cluster_threshold=10)

#########################################################################
# Now use FDR <.05, no cluster-level threshold
thresholded_map2, threshold2 = map_threshold(
    z_map, threshold=.05, height_control='fdr')

#########################################################################
# Visualize the results
from nilearn import plotting
display = plotting.plot_stat_map(z_map, title='Raw z map')
plotting.plot_stat_map(
    thresholded_map1, cut_coords=display.cut_coords, threshold=threshold1,
    title='Thresholded z map, fpr <.001, clusters > 10 voxels')
plotting.plot_stat_map(thresholded_map2, cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fdr = .05',
                       threshold=threshold2)

plotting.show()
