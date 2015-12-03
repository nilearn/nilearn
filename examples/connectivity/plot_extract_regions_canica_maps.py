"""
Regions extraction using Canonical ICA maps and functional connectomes
======================================================================

This example shows how to use :class:`nilearn.regions.RegionExtractor`
to extract connected brain regions from whole brain ICA maps and
use them to estimate a connectome.

We used 20 resting state ADHD functional datasets from :func:`nilearn.datasets.fetch_adhd`
and :class:`nilearn.decomposition.CanICA` for whole brain ICA maps.

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
# Canonical ICA decomposition of functional datasets by importing CanICA from
# decomposition module
from nilearn.decomposition import CanICA

# Initialize canica parameters
canica = CanICA(n_components=5, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                random_state=0)
# Fit to the data
canica.fit(func_filenames)
# ICA maps
components_img = canica.masker_.inverse_transform(canica.components_)

# Visualization
# Show ICA maps by using plotting utilities
from nilearn import plotting

plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='ICA components')

################################################################################
# Extracting regions from ICA maps and then timeseries signals from those
# regions, both can be done by importing Region Extractor from regions module.
# threshold=0.5 indicates that we keep nominal of amount nonzero voxels across all
# maps, less the threshold means that more intense non-voxels will be survived.
from nilearn.regions import RegionExtractor

extractor = RegionExtractor(components_img, threshold=0.5,
                            thresholding_strategy='ratio_n_voxels',
                            extractor='local_regions',
                            standardize=True, min_region_size=50)
# Just call fit() to process for regions extraction
extractor.fit()
# Extracted regions are stored in regions_img_
regions_extracted_img = extractor.regions_img_
# Total number of regions extracted
n_regions_extracted = regions_extracted_img.shape[-1]

# Visualization
# Show region extraction results
title = ('%d regions are extracted from %d ICA components.'
         '\nEach separate color of region indicates extracted region'
         % (n_regions_extracted, 5))
plotting.plot_prob_atlas(regions_extracted_img, view_type='filled_contours',
                         title=title)

################################################################################
# Computing correlation coefficients
# First we need to do subjects timeseries signals extraction and then estimating
# correlation matrices on those signals.
# To extract timeseries signals, we call transform() from RegionExtractor object
# onto each subject functional data stored in func_filenames.
# To estimate correlation matrices we import connectome utilities from nilearn
from nilearn.connectome import ConnectivityMeasure, sym_to_vec

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
# Showing mean correlation results
# Import image utilities in utilising to operate on 4th dimension
import numpy as np
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
# Showing only Default Mode Network (DMN) regions before and after region extraction
# by manually identifying the index of DMN in ICA decomposed components

# First we plot DMN without region extraction, interested in only index=[3]
img = image.index_img(components_img, 3)
coords = plotting.find_xyz_cut_coords(img)
display = plotting.plot_anat(cut_coords=((0, -52, 29)), title='ICA map: DMN')
display.add_contours(img, levels=[0.007], linewidths=2.5, colors=[[0., 0.092711, 1., 1.]],
                     filled=True, linestyles='solid', alpha=0.5)

# Now, we plot DMN after region extraction to show that connected regions are
# nicely separated. Each brain extracted region is indicated with separate color
extracted_DMN_indices = [10, 14, 15, 17]
color_list = [[0., 1., 0.29, 1.], [0., 0.73, 1., 1.],
              [0., 0.47, 1., 1.], [0., 0.22, 1., 1.]]
display = plotting.plot_anat(cut_coords=(0, -52, 29), title='Extracted brain DMN')
# Right Temporoparietal junction
img_rtpj = image.index_img(extractor.regions_img_, extracted_DMN_indices[0])
display.add_contours(img_rtpj, levels=[0.007], linewidths=2.5, colors=[color_list[0]],
                     filled=True, linestyles='solid', alpha=0.5)

# Posterior Cingulate Cortex
img_pcc = image.index_img(extractor.regions_img_, extracted_DMN_indices[1])
display.add_contours(img_pcc, levels=[0.007], linewidths=2.5, colors=[color_list[1]],
                     filled=True, linestyles='solid', alpha=0.5)

# Medial Prefrontal Cortex
img_mpfc = image.index_img(extractor.regions_img_, extracted_DMN_indices[2])
display.add_contours(img_mpfc, levels=[0.007], linewidths=2.5, colors=[color_list[2]],
                     filled=True, linestyles='solid', alpha=0.5)

# Left Temporoparietal junction
img_ltpj = image.index_img(extractor.regions_img_, extracted_DMN_indices[3])
display.add_contours(img_ltpj, levels=[0.007], linewidths=2.5, colors=[color_list[3]],
                     filled=True, linestyles='solid', alpha=0.5)

plotting.show()
