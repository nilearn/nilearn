"""
Regions extraction using Canonical ICA maps
===========================================

This example specifically shows how to segment each ICA map (a 4D Nifti
image/object) into a distinct seperated brain region and extracts timeseries
signals from each seperated region. Both can be done at the same time
using `RegionExtractor`.

We used 40 resting state functional datasets and `CanICA` for 4D components.

This example motivates, how to use `RegionExtractor` to study
functional connectomes using correlation and partial correlation
matrices.
Please see the related documentation of `RegionExtractor` for more details.
"""
import numpy as np
from nilearn import datasets
print " -- Fetching ADHD resting state functional datasets -- "
adhd_dataset = datasets.fetch_adhd(n_subjects=40)
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

from nilearn.input_data import NiftiMasker
print " -- Computing the mask from the data-- "
func_filename = func_filenames[0]
masker = NiftiMasker(standardize=False, mask_strategy='epi')
masker.fit(func_filename)
mask_img = masker.mask_img_

from nilearn.decomposition.canica import CanICA
print " -- Canonical ICA decomposition of functional datasets -- "
# Initialize canica parameters
n_components = 20
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                threshold=3., random_state=0)

canica.fit(func_filenames)
components_img = canica.masker_.inverse_transform(canica.components_)

# Segementation step: Region extraction from ICA maps
# Signals step: Average timeseries signal extraction
# Both are done by calling fit_transform()
from nilearn.regions import region_extractor
print " -- Extracting regions from ICA maps and timeseries signals -- "
reg_ext = region_extractor.RegionExtractor(components_img,
                                           standardize=True,
                                           threshold=0.5, min_size=300,
                                           threshold_strategy='ratio_n_voxels',
                                           extractor='local_regions')
reg_ext.fit_transform(func_filenames, confounds=confounds)
# Regions extracted
regions_extracted_from_ica = reg_ext.regions_
n_regions = regions_extracted_from_ica.shape[3]
print " ====== Regions extracted ====== "
print " -- Number of regions extracted from %d ICA components are %d -- " % (
    n_components, n_regions)
# Index of each region to identify its corresponding ICA map
index_of_each_extracted_region = reg_ext.index_
# Timeseries signals extracted from all subjects
subjects_timeseries = reg_ext.signals_

# Computing correlation coefficients of the timeseries signals
correlation = []
for each_signal in subjects_timeseries:
    corr = np.corrcoef(each_signal.T)
    correlation.append(corr)

# Show the results
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import iter_img
regions_imgs = iter_img(regions_extracted_from_ica)
coords = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
# Show ICA results
plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='ICA components')
# Show region extraction results
plotting.plot_prob_atlas(regions_extracted_from_ica, view_type='filled_contours',
                         title='Regions extracted from ICA components.'
                         ' \nEach color identifies a segmented region')

# Show mean of correlation and partial correlation matrices
title = 'Correlation coefficients showing for %d regions' % n_regions
plt.figure()
plt.imshow(np.mean(correlation, axis=0), interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(np.mean(correlation, axis=0),
                         coords, edge_threshold='90%',
                         title='Correlation')
plt.show()
