"""
Regions Extraction of Default Mode Networks using Smith Atlas
=============================================================

This example shows how to extract regions from Smith Atlas resting
state networks.
"""
# Fetch the datasets and atlas maps
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.image import new_img_like
from nilearn.plotting import find_xyz_cut_coords
print (" -- Fetching Smith Atlas ICA Networks -- ")
smith_atlas = datasets.fetch_atlas_smith_2009()
atlas_networks = smith_atlas.rsn10

# Region Extraction
from nilearn.regions import region_extractor
print (" -- Extracting Networks -- ")
extraction = region_extractor.RegionExtractor(atlas_networks, threshold="98%",
                                              min_size=100)
extraction.fit()
regions = extraction.regions_

# Visualize the region extraction results
import matplotlib.pyplot as plt
from nilearn import plotting
print (" -- Showing the region extraction results from a Smith RSN -- ")
for i, cur_img in zip(extraction.index_, iter_img(regions)):
    coords = find_xyz_cut_coords(cur_img)
    plotting.plot_stat_map(cur_img, display_mode='z', cut_coords=coords[2:3],
                           title="Region extracted corresponds to Network of %d " % i)

fig = plt.figure(figsize=(5, 5))
plotting.plot_prob_atlas(regions, display_mode='z', figure=fig,
                         cut_coords=1, view_type='contours',
                         title="Regions extracted. Each color is a seperate blob")
plt.show()
