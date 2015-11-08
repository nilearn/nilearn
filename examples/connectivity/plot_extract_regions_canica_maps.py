"""
Regions extraction using Canonical ICA maps and functional connectomes
======================================================================

This example shows how to use :class:`nilearn.regions.region_extractor.RegionExtractor`
to extract connected brain regions from whole brain ICA maps and
use them to estimate a connectome.

We used 20 resting state ADHD functional datasets from :func:`nilearn.datasets.fetch_adhd`
and :class:`nilearn.decomposition.canica` for whole brain ICA maps.

Please see the related documentation of :class:`nilearn.regions.region_extractor.RegionExtractor`
for more details.
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
n_components = 5
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=2,
                threshold=3., random_state=0)

canica.fit(func_filenames)
components_img = canica.masker_.inverse_transform(canica.components_)

from nilearn.regions import region_extractor
print(" -- Extracting regions from ICA maps and timeseries signals -- ")
extractor = region_extractor.RegionExtractor(
    components_img, threshold="98%", standardize=True, min_size=100)
# Regions extraction from ICA maps
extractor.fit()
regions_extracted = extractor.regions_
n_regions = regions_extracted.shape[3]
print(" ====== Regions extracted ====== ")
print(" -- Number of regions extracted from %d ICA components are %d -- " % (
    n_components, n_regions))

# Subjects timeseries signals extraction
subjects_timeseries = []
for img, confound in zip(func_filenames, confounds):
    signals = extractor.transform(img, confounds=confound)
    subjects_timeseries.append(signals)

# Computing correlation coefficients of the timeseries signals
correlation = []
for subject_ts in subjects_timeseries:
    corr = np.corrcoef(subject_ts.T)
    correlation.append(corr)

# Show the results
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import iter_img
regions_imgs = iter_img(regions_extracted)
coords = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
# Show ICA results
fig1 = plt.figure(figsize=(10, 5))
plotting.plot_prob_atlas(components_img, view_type='filled_contours',
                         figure=fig1, title='ICA components')
# Show region extraction results
fig2 = plt.figure(figsize=(10, 5))
plotting.plot_prob_atlas(regions_extracted, view_type='filled_contours',
                         title='Regions extracted from ICA components.'
                         ' \nEach color identifies a segmented region',
                         figure=fig2)

# Show mean of correlation and partial correlation matrices
title = 'Correlation coefficients showing for %d regions' % n_regions
plt.figure()
plt.imshow(np.mean(correlation, axis=0), interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.bwr)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(np.mean(correlation, axis=0),
                         coords, edge_threshold='90%',
                         title='Correlation')
plt.show()
