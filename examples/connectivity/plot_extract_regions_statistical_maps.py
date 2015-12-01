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

################################################################################
# Threshold the t-statistic image by importing threshold function
from nilearn.image import threshold_img

# Two types of strategies can be used from this threshold function
# Type 1: strategy = 'percentile'
threshold_percentile_img = threshold_img(localizer_path.tmaps[2], threshold='95%',
                                         thresholding_strategy='percentile')

# Type 2: strategy = 'img_value'
threshold_value_img = threshold_img(localizer_path.tmaps[2], threshold=4.,
                                    thresholding_strategy='img_value')

################################################################################
# Extracting the regions by importing connected regions function
from nilearn.regions import connected_regions

regions_img_percentile, index = connected_regions(threshold_percentile_img,
                                                  min_region_size=200)

regions_img_value, index = connected_regions(threshold_value_img,
                                             min_region_size=200)

################################################################################
# Visualizing region extraction results by importing plotting tools
from nilearn import plotting

# Visualizing input statistical image
plotting.plot_stat_map(localizer_path.tmaps[2], title='Input data: Statistical t-map')
# Visualizing thresholding results
thresholding_results = {
    'Statistical t-map thresholded using percentile': threshold_percentile_img,
    'Statistical t-map thresholded using image value': threshold_value_img
    }
for title, result in sorted(thresholding_results.items()):
    plotting.plot_stat_map(result, title=title)

# Visualizing region extraction results
title = ("Region Extraction results on 'percentile' thresholded image. "
         "\n Each color indicates segmented region")
plotting.plot_prob_atlas(regions_img_percentile, anat_img=localizer_path.tmaps[2],
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
title = ("Region Extraction results on 'img_value' thresholded image. "
         "\n Each color indicates segmented region")
plotting.plot_prob_atlas(regions_img_value, anat_img=localizer_path.tmaps[2],
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
plotting.show()
