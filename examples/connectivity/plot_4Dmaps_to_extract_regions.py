"""
Regions and Timeseries signal extraction using Canonical ICA maps
=================================================================

This example specifically shows how to segment each ICA map (a 4D Nifti
image/object) into a distinct seperated brain region and extracts timeseries
signals from each seperated region. Both can be done at the same time
using module "region_extractor".
Please see the related documentation for more details.

This example is also motivated to show how to use module "region_extractor"
to study functional connectomes using correlation and partial correlation
matrices and finally displaying the classification scores predictions between
ADHD and healthy subjects. For this, we used a total of 40 resting state
functional datasets which has both ADHD and healthy categories.
"""
import numpy as np


def cov_to_corr(covariance_matrix):
    """ Return correlation matrix for a given covariance matrix. """
    diag = np.atleast_2d(1. / np.sqrt(np.diag(covariance_matrix)))
    correlation = covariance_matrix * diag * diag.T
    return correlation


def prec_to_partial(precision_matrix):
    """Return partial correlation for a given precision matrix. """
    partial_correlation = -cov_to_corr(precision_matrix)
    np.fill_diagonal(partial_correlation, 1.)
    return partial_correlation


def triu(matrix, k=0):
    """ Returns the upper traingle as a row matrix for given input matrix. """
    assert(matrix.shape[0] == matrix.shape[1])
    return matrix[np.triu_indices(matrix.shape[0], k=k)]


from nilearn import datasets
print "-- Fetching ADHD resting state functional datasets --"
adhd_dataset = datasets.fetch_adhd()
func_filenames = adhd_dataset.func
confounds = adhd_dataset.confounds

from nilearn.input_data import NiftiMasker
print "-- Computing the mask from the data--"
func_filename = func_filenames[0]
masker = NiftiMasker(standardize=False, mask_strategy='epi')
masker.fit(func_filename)
mask_img = masker.mask_img_

from nilearn.decomposition.canica import CanICA
print "-- Canonical ICA decomposition of functional datasets --"
# Initialize canica parameters
n_components = 10
canica = CanICA(n_components=n_components, smoothing_fwhm=6.,
                memory="nilearn_cache", memory_level=5,
                threshold=3., random_state=0)

canica.fit(func_filenames)
components_img = canica.masker_.inverse_transform(canica.components_)

# Segementation step: Region extraction from ICA maps
# Signals step: Average timeseries signal extraction
# Both are done by calling fit_transform()
from nilearn.region_decomposition import region_extractor
print "-- Extracting regions from ICA maps and timeseries signals --"
reg_ext = region_extractor.RegionExtractor(components_img,
                                           threshold=0.5, min_size=200,
                                           threshold_strategy='percentile',
                                           extractor='local_regions')
reg_ext.fit_transform(func_filenames, confounds=confounds)
# Regions extracted
regions_extracted_from_ica = reg_ext.regions_
n_regions = regions_extracted_from_ica.shape[3]
print "====== Regions extracted ======"
print "-- Number of regions extracted from %d ICA components are %d--" % (
    n_components, n_regions)
# Index of each region to identify its corresponding ICA map
index_of_each_extracted_region = reg_ext.index_
# Timeseries signals extracted from all subjects
subjects_timeseries = reg_ext.signals_

# Estimate correlation and partial correlation matrices using LedoitWolf
# Estimator
from sklearn.covariance import LedoitWolf
l_w = LedoitWolf()
correlation_matrices = []
partial_correlation_matrices = []
correlation = []
partial_correlation = []
# Compute correlation and partial correlation matrices on a single
# subject timeseries level and finally append all matrices to one array
print "-- Ledoit Wolf Shrinkage Estimator --"
for id in range(len(subjects_timeseries)):
    l_w.fit(subjects_timeseries[id])
    # Covariance to Correlation matrix
    print "-- Computing correlation and partial correlation matrix for subject %d --" % id
    corr_matrix = cov_to_corr(l_w.covariance_)
    correlation.append(corr_matrix)
    corr_matrix = triu(corr_matrix, k=1)
    correlation_matrices.append(corr_matrix)
    # Precision to partial correlation matrix
    partial_corr_matrix = prec_to_partial(l_w.precision_)
    partial_correlation.append(partial_corr_matrix)
    partial_corr_matrix = triu(partial_corr_matrix, k=1)
    partial_correlation_matrices.append(partial_corr_matrix)

correlation_matrices = np.asarray(correlation_matrices)
partial_correlation_matrices = np.asarray(partial_correlation_matrices)

# Support Vector Classification
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
print "-- Support Vector Linear Classification between ADHD vs Controls --"
print "-- Classification based on partial correlation matrices --"
X = partial_correlation_matrices
y = adhd_dataset.phenotypic['adhd']

# Stratified Shuffle Split
sss = StratifiedShuffleSplit(y, n_iter=100, test_size=0.25, random_state=0)
svc = SVC(kernel='linear', class_weight='auto', random_state=0)
# Cross Validation Scores
print "-- Computing classification scores --"
cvs = cross_val_score(svc, X, y, scoring='roc_auc', cv=sss)

# Show the results
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.image import iter_img
regions_imgs = iter_img(regions_extracted_from_ica)
coords = [plotting.find_xyz_cut_coords(img) for img in regions_imgs]
# Show ICA results
plotting.plot_prob_atlas(components_img, title='ICA components')
# Show region extraction results
plotting.plot_prob_atlas(regions_extracted_from_ica,
                         title='Regions extracted from ICA components.'
                         ' \nEach color identifies a segmented region')
# Show mean of correlation and partial correlation matrices
title = 'Correlation matrices showing for %d regions' % n_regions
plt.figure()
plt.imshow(np.mean(correlation, axis=0), interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(np.mean(correlation, axis=0),
                         coords, edge_threshold='90%',
                         title='Correlation')
title = 'Partial correlation matrices showing for %d regions' % n_regions
plt.figure()
plt.imshow(np.mean(partial_correlation, axis=0), interpolation="nearest",
           vmax=1, vmin=-1, cmap=plt.cm.coolwarm)
plt.colorbar()
plt.title(title)
plotting.plot_connectome(np.mean(partial_correlation, axis=0),
                         coords, edge_threshold='90%',
                         title='Partial Correlation')
plt.show()
print " ===== ROC AUC Scores ====="
print ("Classification Accuracy: %0.2f (+/- %0.2f)" % (cvs.mean(), cvs.std()))
