"""
Regions extraction using Canonical ICA maps
===========================================

This example shows how to extract each connected ICA map (mostly a 4D Nifti
image/object) into a seperate brain activation regions and extracts timeseries
signals from each seperated region. Both can be done at the same time
using `RegionExtractor`.

We used 20 resting state functional datasets and `CanICA` for 4D components.

This example in particular can be used to study functional connectomes using
correlation network matrices between regions.

Please see the related documentation of `RegionExtractor` for more details.
"""
import numpy as np
from nilearn import datasets
print(" -- Fetching ADHD resting state functional datasets -- ")
adhd_dataset = datasets.fetch_adhd(n_subjects=20)
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

from nilearn.decomposition.canica import CanICA
print(" -- Canonical ICA decomposition of functional datasets -- ")
# Initialize canica parameters
n_components = 20
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=1,
                threshold=3., random_state=0)

canica.fit(func_filenames)
components_img = canica.masker_.inverse_transform(canica.components_)

# Segementation step: Region extraction from ICA maps
# Signals step: Average timeseries signal extraction
# Both are done by calling fit_transform()
from nilearn.regions import region_extractor
print(" -- Extracting regions from ICA maps and timeseries signals -- ")
extractor = region_extractor.RegionExtractor(
    components_img, standardize=True, threshold=0.3, min_size=300,
    thresholding_strategy='ratio_n_voxels', extractor='local_regions')
extractor.fit_transform(func_filenames, confounds=confounds)

regions_extracted = extractor.regions_
n_regions = regions_extracted.shape[3]
print(" ====== Regions extracted ====== ")
print(" -- Number of regions extracted from %d ICA components are %d -- " % (
    n_components, n_regions))

# Timeseries signals extracted from all subjects
subjects_timeseries = extractor.signals_

# Computing correlation coefficients of the timeseries signals
correlation = []
for each_signal in subjects_timeseries:
    corr = np.corrcoef(each_signal.T)
    correlation.append(corr)

# Show the results
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import iter_img
regions_imgs = iter_img(regions_extracted)
coords = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
# Show ICA results
plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         title='ICA components')
# Show region extraction results
plotting.plot_prob_atlas(regions_extracted, view_type='filled_contours',
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
