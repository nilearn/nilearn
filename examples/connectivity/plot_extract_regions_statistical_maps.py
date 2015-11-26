"""
Region Extraction using a t-statistical map (3D)
================================================

This example shows how to extract regions or seperate the regions
from a statistical map.

We use localizer t-statistic maps from :func:`nilearn.datasets.fetch_localizer_contrasts`
as an input image.

The idea is to threshold an image to get foreground objects using a
function :func:`nilearn.image.threshold_img` and extract objects using a function
:func:`nilearn.regions.connected_regions`.
"""

# Load localizer datasets - contrast/t maps
from nilearn import datasets
print(" -- Fetching t-statistic image from localizer datasets -- ")
n_subjects = 3
localizer_path = datasets.fetch_localizer_contrasts(
    ['calculation (auditory cue)'], n_subjects=n_subjects, get_tmaps=True)

from nilearn.image import threshold_img
print(" -- Thresholding -- ")
threshold_strategies = ['percentile', 'img_value']
threshold_value = ['95%', 4.]
thresholding = {}
for thr, strategy in zip(threshold_value, threshold_strategies):
    thresholding[strategy] = threshold_img(localizer_path.tmaps[2],
                                           threshold=thr,
                                           thresholding_strategy=strategy)

from nilearn.regions import connected_regions
print(" -- Regions Extraction -- ")
regions = []
for strategy in threshold_strategies:
    region, _ = connected_regions(thresholding[strategy], min_region_size=200)
    regions.append(region)

# Visualization
import matplotlib.pyplot as plt
from nilearn import plotting

print(" -- Visualizing input statistical image -- ")
plotting.plot_stat_map(localizer_path.tmaps[2], title='Input data: Statistical t-map')

print(" -- Visualizing thresholding results -- ")
thresholding_results = {
    'Statistical t-map thresholded using percentile': thresholding['percentile'],
    'Statistical t-map thresholded using image value': thresholding['img_value']
    }
for title, result in sorted(thresholding_results.items()):
    plotting.plot_stat_map(result, title=title)

print(" -- Visualizing region extraction results -- ")
title = ('Regions Extraction results on percentile thresholded image'
         '\n Each color indicates segmented region')
plotting.plot_prob_atlas(regions[0], anat_img=localizer_path.tmaps[2],
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
title = ('Regions Extraction results on image value thresholded image'
         '\n Each color indicates segmented region')
plotting.plot_prob_atlas(regions[1], anat_img=localizer_path.tmaps[2],
                         view_type='contours', display_mode='z',
                         cut_coords=5, title=title)
plt.show()
