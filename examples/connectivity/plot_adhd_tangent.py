"""
Comparing different connectivity measures
=========================================

This example compares different measures of functional connectivity between
regions of interest.

"""

# Fetch dataset
import nilearn.datasets
atlas = nilearn.datasets.fetch_atlas_msdl()
dataset = nilearn.datasets.fetch_adhd()

# Extract regions time series
import nilearn.input_data
masker = nilearn.input_data.NiftiMapsMasker(
    atlas.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
    memory='nilearn_cache', memory_level=1)
subjects = []
for func_file in dataset.func:
    time_series = masker.fit_transform(func_file)
    subjects.append(time_series)

# Estimate connectivity matrices
import nilearn.connectivity
subjects_connectivity = {}
measures = ['tangent', 'partial correlation', 'precision', 'correlation',
            'covariance']
mean_connectivity = {}
from sklearn.covariance import EmpiricalCovariance
for measure in measures:
    estimator = {'measure': measure, 'cov_estimator': EmpiricalCovariance()}
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    subjects_connectivity[measure] = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    # Compute the mean connectivity across all subjects
    if measure == 'tangent':
        mean_connectivity[measure] = cov_embedding.tangent_mean_
    else:
        mean_connectivity[measure] = \
            subjects_connectivity[measure].mean(axis=0)

# Plot the mean connectivity
import numpy as np
import nilearn.plotting
labels = np.recfromcsv(atlas.labels)
region_coords = np.vstack((labels['x'], labels['y'], labels['z'])).T
for measure in ['tangent', 'correlation', 'partial correlation']:
    nilearn.plotting.plot_connectome(mean_connectivity[measure], region_coords,
                                     edge_threshold='98%',
                                     title='mean %s' % measure)

# Plot a connectivity matrix for one subject
import matplotlib.pyplot as plt
subject_n = 28
plt.figure()
plt.imshow(subjects_connectivity['correlation'][subject_n],
           interpolation="nearest", vmin=-1., vmax=1.)
plt.title('subject %d, correlation' % subject_n)

# Get site and ADHD/control label for each subject
adhds = dataset.phenotypic['adhd']
sites = ['"Peking"' if 'Peking' in site else site for site in
         dataset.phenotypic['site']]  # Group Peking sites

# Use connectivity coefficients to classify ADHD vs controls
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
classes = [site + str(adhd) for site, adhd in zip(sites, adhds)]
cv = StratifiedShuffleSplit(classes, n_iter=1000, test_size=0.33)
mean_scores = []
print('Classification accuracy:')
for measure in measures:
    svc = LinearSVC()
    coefs_vec = nilearn.connectivity.embedding.sym_to_vec(
        subjects_connectivity[measure])
    cv_scores = cross_val_score(
        svc, coefs_vec, adhds, cv=cv, scoring='accuracy')
    print('%20s score: %1.2f +- %1.2f' % (measure, cv_scores.mean(),
                                          cv_scores.std()))
    mean_scores.append(cv_scores.mean())

# Display the classification scores
plt.figure()
positions = np.arange(len(measures)) * .5 + .5
plt.barh(positions, mean_scores, align='center', height=0.2)
measures[1] = 'partial\ncorrelation'
plt.yticks(positions, measures)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.show()
