"""
Regions extraction using Dictionary Learning and functional connectomes
=======================================================================

This example shows how to use :class:`nilearn.regions.RegionExtractor`
to extract spatially constrained brain regions from whole brain maps decomposed
using dictionary learning and use them to build a functional connectome.

We used 20 resting state ADHD functional datasets from :func:`nilearn.datasets.fetch_adhd`
and :class:`nilearn.decomposition.DictLearning` for set of brain atlas maps.

This example can also be inspired to apply the same steps to even regions extraction
using ICA maps. In that case, idea would be to replace dictionary learning to canonical
ICA decomposition using :class:`nilearn.decomposition.CanICA`

Please see the related documentation of :class:`nilearn.regions.RegionExtractor`
for more details.
"""

################################################################################
# Fetching ADHD resting state functional datasets by loading from datasets
# utilities
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=20)
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

################################################################################
# Extracting resting-state networks with DictionaryLearning

# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=5, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          random_state=0)
# Fit to the data
dict_learn.fit(func_filenames)
# Resting state networks/maps
components_img = dict_learn.masker_.inverse_transform(dict_learn.components_)

# Visualization of resting state networks
# Show networks using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')

################################################################################
# Extracting regions from networks

# Import Region Extractor algorithm from regions module
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=1350)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Each region index is stored in index_
regions_index = extractor.index_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization of region extraction results
title = ('%d regions are extracted from %d components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 5))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)

################################################################################
# Computing correlation coefficients and plotting a connectome

# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn
from nilearn.connectome import ConnectivityMeasure

correlations = []
# Initializing ConnectivityMeasure object with kind='correlation'
connectome_measure = ConnectivityMeasure(kind='correlation')
for filename, confound in zip(func_filenames, confounds):
    # call transform from RegionExtractor object to extract timeseries signals
    timeseries_each_subject = extractor.transform(filename, confounds=confound)
    # call fit_transform from ConnectivityMeasure object
    correlation = connectome_measure.fit_transform([timeseries_each_subject])
    # saving each subject correlation to correlations
    correlations.append(correlation)

# Mean of all correlations
import numpy as np

mean_correlations = np.mean(correlations, axis=0).reshape(n_regions_extracted,
                                                          n_regions_extracted)

# Visualization
# Plotting connectome results
import matplotlib.pyplot as plt
from nilearn import image

regions_imgs = image.iter_img(regions_extracted_img)
coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
title = 'Correlation interactions between %d regions' % n_regions_extracted
plt.figure()
plt.imshow(mean_correlations, interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.bwr)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)

################################################################################
# Plotting regions extracted for only one specific network

# First, we plot a network of index=4 without region extraction (left plot)
img = image.index_img(components_img, 4)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords,
                                 colorbar=False, title='Showing one specific network')

# Now, we plot (right side) same network after region extraction to show that
# connected regions are nicely seperated.
# Each brain extracted region is identified as separate color.

# For this, we take the indices of the all regions extracted related to original
# network given as 4.
regions_indices_of_map3 = np.where(np.array(regions_index) == 4)

display = plotting.plot_anat(cut_coords=coords,
                             title='Extracted regions in one specific network')

# Now add as an overlay by looping over all the regions of index 4
# color list is random (you can choose your own color)
color_list = [[0., 1., 0.29, 1.], [0., 1., 0.54, 1.],
              [0., 1., 0.78, 1.], [0., 0.96, 1., 1.],
              [0., 0.73, 1., 1.], [0., 0.47, 1., 1.],
              [0., 0.22, 1., 1.], [0.01, 0., 1., 1.],
              [0.26, 0., 1., 1.]]
for each_index_of_map3, color in zip(regions_indices_of_map3[0], color_list):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

plotting.show()
