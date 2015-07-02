"""
Comparing different connectivity measures
=========================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate different connectivity measures based on these signals.
"""

import matplotlib.pyplot as plt
import numpy as np

print("-- Fetching datasets ...")
import nilearn.datasets
atlas = nilearn.datasets.fetch_msdl_atlas()
dataset = nilearn.datasets.fetch_adhd()

import nilearn.image
import nilearn.input_data

import joblib
mem = joblib.Memory("/home/sb238920/CODE/Parietal/nilearn/nilearn_cache/adhd")

# Number of subjects to consider
n_subjects = 40
subjects = []
for subject_n in range(n_subjects):
    filename = dataset["func"][subject_n]
    print("Processing file %s" % filename)

    print("-- Computing confounds ...")
    confound_file = dataset["confounds"][subject_n]
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(filename)

    print("-- Computing region signals ...")
    masker = nilearn.input_data.NiftiMapsMasker(
        atlas["maps"], resampling_target="maps", detrend=True,
        low_pass=None, high_pass=0.01, t_r=2.5, standardize=False,
        memory=mem, memory_level=1, verbose=1)
    region_ts = masker.fit_transform(filename,
                                     confounds=[hv_confounds, confound_file])
    subjects.append(region_ts)


import nilearn.connectivity
all_matrices = []
measures = ['correlation', 'partial correlation', 'tangent', 'covariance',
            'precision']
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
sites = np.array([k / 8 for k in range(n_subjects)])
adhd = dataset.phenotypic['adhd'][:n_subjects]
subjects = np.array(subjects)

# Define parameters
sites_gmean = False  # if True, gmean is computed sperately for each site
adhd_gmean = False  # if True, gmean is computed sperately for ADHD/controls
cov_estimator = EmpiricalCovariance()
print('specific gmean for each site: {0}, '
      'specific gmean for each ADHD/controls: {1}'
      'estimator: {3}'.format(sites_gmean, adhd_gmean, cov_estimator))
for kind in measures:
    estimator = {'kind': kind, 'cov_estimator': cov_estimator}
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    matrices = np.zeros((n_subjects, region_ts.shape[-1], region_ts.shape[-1]))
    if sites_gmean:
        for n_site in range(0, n_subjects / 8):
            matrices[n_site * 8:(n_site + 1) * 8] = \
                nilearn.connectivity.vec_to_sym(cov_embedding.fit_transform(
                                    subjects[n_site * 8:(n_site + 1) * 8]))
    elif adhd_gmean:
        matrices[adhd == 1] = nilearn.connectivity.vec_to_sym(
            cov_embedding.fit_transform(
                subjects[adhd == 1]))
        matrices[adhd == 0] = nilearn.connectivity.vec_to_sym(
            cov_embedding.fit_transform(
                subjects[adhd == 0]))
    else:
        matrices = nilearn.connectivity.vec_to_sym(
            cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)


# Classify sites and ADHD
from sklearn.svm import LinearSVC
from sklearn.lda import LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
both = sites * 2 + adhd
clfs = [LinearSVC(), LDA()] + [KNeighborsClassifier(n_neighbors=n_neighbors)
                               for n_neighbors in range(1, 6)]
clf_names = ['SVM', 'LDA'] + ['KNN n={}'.format(n) for n in range(1, 6)]
cv = StratifiedShuffleSplit(both, n_iter=10000, test_size=0.33)
for classes, prediction in zip([sites, adhd], ['sites', 'ADHD/controls']):
    print('-- {} classification ...'.format(prediction))
    scores = {}
    for measure, coefs in zip(measures, all_matrices):
        scores[measure] = {}
        for clf, clf_name in zip(clfs, clf_names):
            coefs_vec = nilearn.connectivity.embedding.sym_to_vec(coefs)
            scores[measure][clf_name] = cross_val_score(
                clf, coefs_vec, classes, cv=cv, scoring='accuracy')
            print(' {0}, classifier {1}: score is {2:.2f} +- {3:.2f}'.format(
                  measure, clf_name, scores[measure][clf_name].mean(),
                  scores[measure][clf_name].std()))

    plt.figure()
    tick_position = np.arange(len(clfs))
    plt.xticks(tick_position + 0.4, clf_names)
    for color, measure in zip('kcgbr', measures):
        score_means = [scores[measure][clf_name].mean() for clf_name in
                       clf_names]
        score_stds = [scores[measure][clf_name].std() for clf_name in
                      clf_names]
        plt.bar(tick_position, score_means, yerr=score_stds, label=measure,
                color=color, width=.2)
        tick_position = tick_position + .15
    plt.ylabel('Classification accuracy')
    plt.legend(measures)
    plt.title(prediction)

plt.show()
