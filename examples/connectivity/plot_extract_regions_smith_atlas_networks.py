"""
Regions Extraction of Default Mode Networks using Smith Atlas
=============================================================

This example shows how to extract regions within default mode
networks DMN using ICA maps of Smith Atlas.

This is done by visually identifying the DMN index and using
`index_img` to get an image.
"""
# Fetch the datasets and atlas maps
from nilearn import datasets
from nilearn.image import iter_img
from nilearn.image import new_img_like
from nilearn.plotting import find_xyz_cut_coords
print (" -- Fetching Smith Atlas ICA Networks -- ")
smith_atlas = datasets.fetch_atlas_smith_2009()
atlas_networks = smith_atlas.rsn10

adhd_dataset = datasets.fetch_adhd(n_subjects=1)
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

# Region Extraction
from nilearn.regions import region_extractor
print (" -- Extracting Default Mode Networks -- ")
extraction = region_extractor.RegionExtractor(atlas_networks, threshold=0.5,
                                              thresholding_strategy='percentile',
                                              extractor='local_regions',
                                              min_size=200)
extraction.fit_transform(func_filenames, confounds=confounds)

regions = extraction.regions_
index = extraction.index_

# Visualize the region extraction results
import matplotlib.pyplot as plt
from nilearn import plotting
print (" -- Showing the region extraction results from a Smith RSN -- ")
for i, cur_img in zip(index, iter_img(regions)):
    coords = find_xyz_cut_coords(cur_img)
    data = cur_img.get_data()
    data.shape = (cur_img.get_data()).shape + (1, )
    img = new_img_like(cur_img, data, cur_img.get_affine())
    plotting.plot_prob_atlas(img, display_mode='z', cut_coords=coords[2:3],
                             view_type='contours',
                             title="Region extracted corresponds to Network of %d " % i)

plotting.plot_prob_atlas(regions, display_mode='z',
                         cut_coords=1, view_type='contours',
                         title="Regions extracted. Each color is a seperate blob")
plt.show()
