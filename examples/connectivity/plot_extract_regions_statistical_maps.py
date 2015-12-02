"""
Region Extraction using a t-statistical map (3D)
================================================

This example shows how to extract regions or separate the regions
from a statistical map.

We use localizer t-statistic maps from :func:`nilearn.datasets.fetch_localizer_contrasts`
as an input image.

The idea is to threshold an image to get foreground objects using a
function :func:`nilearn.image.threshold_img` and extract objects using a function
:func:`nilearn.regions.connected_regions`.
"""

################################################################################
# Fetching t-statistic image of localizer constrasts by loading from datasets
# utilities
from nilearn import datasets

n_subjects = 3
localizer_path = datasets.fetch_localizer_contrasts(
    ['calculation (auditory cue)'], n_subjects=n_subjects, get_tmaps=True)
tmap_filename = localizer_path.tmaps[2]

################################################################################
# Threshold the t-statistic image by importing threshold function
from nilearn.image import threshold_img

# Two types of strategies can be used from this threshold function
# Type 1: strategy = 'percentile'
threshold_percentile_img = threshold_img(tmap_filename, threshold='95%',
                                         thresholding_strategy='percentile')

# Type 2: strategy = 'img_value'
# Here, threshold value should be within the limits i.e. less than max value.
threshold_value_img = threshold_img(tmap_filename, threshold=4.,
                                    thresholding_strategy='img_value')

################################################################################
# Extracting the regions by importing connected regions function
from nilearn.regions import connected_regions

regions_percentile_img, index = connected_regions(threshold_percentile_img,
                                                  min_region_size=100)

regions_value_img, index = connected_regions(threshold_value_img,
                                             min_region_size=100)

################################################################################
# Visualizing region extraction results by importing plotting tools
from nilearn import plotting

# Visualizing input statistical image
plotting.plot_stat_map(tmap_filename, title='Input data: Statistical t-map')

# Visualizing thresholding results
# Showing threshold image thresholded using percentile strategy
plotting.plot_stat_map(threshold_percentile_img,
                       title='Statistical t-map thresholded using percentile')
# Showing threshold image thresholded using image value strategy
plotting.plot_stat_map(threshold_value_img,
                       title='Statistical t-map thresholded using image value')

# Visualizing region extraction results
title = ("Region Extraction results on 'percentile' thresholded image. "
         "\n Each color indicates segmented region")
plotting.plot_prob_atlas(regions_percentile_img, anat_img=tmap_filename,
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
title = ("Region Extraction results on 'img_value' thresholded image. "
         "\n Each color indicates segmented region")
plotting.plot_prob_atlas(regions_value_img, anat_img=tmap_filename,
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
plotting.show()
