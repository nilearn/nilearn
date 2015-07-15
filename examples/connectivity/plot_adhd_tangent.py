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

# Extract and preprocess regions time series
import nilearn.input_data
import joblib
mem = joblib.Memory('/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/adhd')
masker = nilearn.input_data.NiftiMapsMasker(
    atlas.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
    memory=mem, memory_level=1)
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
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RidgeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
classifiers = [LinearSVC(), KNeighborsClassifier(n_neighbors=1),
               LogisticRegression(), GaussianNB(), RidgeClassifier()]
classifier_names = ['SVM', 'KNN', 'logistic', 'GNB', 'ridge']
classes = [site + str(adhd) for site, adhd in zip(sites, adhds)]
cv = StratifiedShuffleSplit(classes, n_iter=1000, test_size=0.33)
scores = {}
for measure in measures:
    scores[measure] = {}
    print('---------- %20s ----------' % measure)
    for classifier, classifier_name in zip(classifiers, classifier_names):
        coefs_vec = nilearn.connectivity.embedding.sym_to_vec(
            subjects_connectivity[measure])
        scores[measure][classifier_name] = cross_val_score(
            classifier, coefs_vec, adhds, cv=cv, scoring='accuracy')
        print(' %14s score: %1.2f +- %1.2f' % (classifier_name,
              scores[measure][classifier_name].mean(),
              scores[measure][classifier_name].std()))

# Display the classification scores
plt.figure()
tick_position = np.arange(len(classifiers))
plt.xticks(tick_position + 0.35, classifier_names)
for color, measure in zip('rgbyk', measures):
    score_means = [scores[measure][classifier_name].mean() for classifier_name
                   in classifier_names]
    plt.bar(tick_position, score_means, label=measure, color=color, width=.2)
    tick_position = tick_position + .15
plt.ylabel('Classification accuracy')
plt.legend(measures, loc='upper left')
plt.ylim([0., 1.])
plt.show()