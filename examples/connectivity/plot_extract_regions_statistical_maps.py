"""
Region Extraction using a t-statistical maps (3D)
=================================================

This example shows how to extract regions or seperate the regions
from the statistical maps.

We use localizer t-statistic maps as an input image.

The idea is to threshold an input for foreground objects using a
function :func:`nilearn.regions.region_extractor.foreground_extraction`
and extract objects using a function :func:`nilearn.regions.region_extractor.connected_component_extraction`.

In this example, we also show how to use raw t-statistical value as a
threshold and extract regions based on survived foreground objects.
"""

# Load localizer datasets - contrast/t maps
from nilearn import datasets
print(" -- Fetching t-statistic image from localizer datasets -- ")
n_subjects = 3
localizer_path = datasets.fetch_localizer_contrasts(
    ['calculation (auditory cue)'], n_subjects=n_subjects, get_tmaps=True)

# Foreground Extraction
from nilearn.regions.region_extractor import foreground_extraction
print(" -- Foreground Extraction -- ")
# Foreground Extraction showing for two different types of strategies.
# Here, we used thresholding_strategy='percentile' and None.
threshold_strategies = ['percentile', None]
threshold_value = [0.2, 4.]
foreground_extracted = {}
for thr, strategy in zip(threshold_value, threshold_strategies):
    foreground_extracted[strategy] = foreground_extraction(
        localizer_path.tmaps[2], threshold=thr, thresholding_strategy=strategy)

# Region extraction
from nilearn.regions.region_extractor import connected_component_extraction
print(" -- Region Extraction -- ")
# Region extraction showing for two different types of extraction methods.
# We used, extract_type="connected_components" and random walker as "local_regions"
region_extractors = ['connected_components', 'local_regions']
regions = {}
for strategy, extractor in zip(threshold_strategies, region_extractors):
    regions[extractor], _ = connected_component_extraction(
        foreground_extracted[strategy], min_size=200, extract_type=extractor)

# Region extraction on a t-statistic value based threshold image.
# extract_type=default
regions_tstat, _ = connected_component_extraction(foreground_extracted[None],
                                                  min_size=200)

# Visualization
import matplotlib.pyplot as plt
from nilearn import plotting
print(" -- Showing the results -- ")

# Visualize foreground extracted results
foreground_extraction_results = {
    'Input_data: Statistical t-map': localizer_path.tmaps[2],
    'Statistical t-map after foreground extraction'
    ' \nusing a percentile strategy': foreground_extracted['percentile'],
    'Statistical t-map after thresholding'
    ' \nusing t-value': foreground_extracted[None]
    }

for title, result in sorted(foreground_extraction_results.items()):
    plotting.plot_stat_map(result, title=title)

# Visualize region extraction results
region_extraction_results = {
    'Regions extracted from a "t value" based threshold'
    ' \nEach color identifies a seperate region': regions_tstat,
    'Regions extracted by an extract type of "connected components"'
    ' \nEach color identifies a seperate region': regions['connected_components'],
    'Regions extracted by an extract type of "random walker"'
    ' \nEach color identifies a seperate region': regions['local_regions']
    }
for title, extract_map in sorted(region_extraction_results.items()):
    plotting.plot_prob_atlas(extract_map, anat_img=localizer_path.tmaps[2],
                             view_type='contours', display_mode='z',
                             cut_coords=5, title=title)

plt.show()
