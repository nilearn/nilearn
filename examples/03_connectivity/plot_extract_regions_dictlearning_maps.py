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

.. note::

    The use of the attribute `components_img_` from dictionary learning
    estimator is implemented from version 0.4.1. For older versions,
    unmask the deprecated attribute `components_` to get the components
    image using attribute `masker_` embedded in estimator.
    See the :ref:`section Inverse transform: unmasking data <unmasking_step>`.
"""

################################################################################
# Fetch ADHD resting state functional datasets
# ---------------------------------------------
#
# We use nilearn's datasets downloading utilities
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=20)
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

################################################################################
# Extract resting-state networks with DictionaryLearning
# -------------------------------------------------------

# Import dictionary learning algorithm from decomposition module and call the
# object and fit the model to the functional datasets
from nilearn.decomposition import DictLearning

# Initialize DictLearning object
dict_learn = DictLearning(n_components=5, smoothing_fwhm=6.,
                          memory="nilearn_cache", memory_level=2,
                          random_state=0)
# Fit to the data
dict_learn.fit(func_filenames)
# Resting state networks/maps in attribute `components_img_`
# Note that this attribute is implemented from version 0.4.1.
# For older versions, see the note section above for details.
components_img = dict_learn.components_img_

# Visualization of resting state networks
# Show networks using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='Dictionary Learning maps')

################################################################################
# Extract regions from networks
# ------------------------------

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
# Compute correlation coefficients
# ---------------------------------

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

###############################################################################
# Plot resulting connectomes
# ----------------------------

title = 'Correlation between %d regions' % n_regions_extracted

# First plot the matrix
display = plotting.plot_matrix(mean_correlations, vmax=1, vmin=-1,
                               colorbar=True, title=title)

# Then find the center of the regions and plot a connectome
from nilearn import image
regions_imgs = image.iter_img(regions_extracted_img)
coords_connectome = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]

plotting.plot_connectome(mean_correlations, coords_connectome,
                         edge_threshold='90%', title=title)

################################################################################
# Plot regions extracted for only one specific network
# ----------------------------------------------------

# First, we plot a network of index=4 without region extraction (left plot)
img = image.index_img(components_img, 4)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_stat_map(img, cut_coords=coords, colorbar=False,
                                 title='Showing one specific network')

################################################################################
# Now, we plot (right side) same network after region extraction to show that
# connected regions are nicely seperated.
# Each brain extracted region is identified as separate color.

# For this, we take the indices of the all regions extracted related to original
# network given as 4.
regions_indices_of_map3 = np.where(np.array(regions_index) == 4)

display = plotting.plot_anat(cut_coords=coords,
                             title='Regions from this network')

# Add as an overlay all the regions of index 4
colors = 'rgbcmyk'
for each_index_of_map3, color in zip(regions_indices_of_map3[0], colors):
    display.add_overlay(image.index_img(regions_extracted_img, each_index_of_map3),
                        cmap=plotting.cm.alpha_cmap(color))

plotting.show()
