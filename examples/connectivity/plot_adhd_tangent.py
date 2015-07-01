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
print("-- Measuring connecivity ...")
all_matrices = []
measures = ['correlation', 'partial correlation', 'tangent', 'covariance',
            'precision']
for kind in measures:
    estimator = {'kind': kind}
    cov_embedding = nilearn.connectivity.CovEmbedding(**estimator)
    matrices = nilearn.connectivity.vec_to_sym(
        cov_embedding.fit_transform(subjects))
    all_matrices.append(matrices)

# Classify sites and patients
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
sites = np.array([k / 8 for k in range(n_subjects)])
adhd = dataset.phenotypic['adhd'][:n_subjects]
both = sites * 2 + adhd
bar_position = 0
for classes in [sites, adhd]:
    for measure, coefs, color in zip(measures, all_matrices, 'kcgbr'):
        coefs_vec = nilearn.connectivity.embedding.sym_to_vec(coefs)
        cv = StratifiedKFold(both, 3)  # stratify on both sites and ADHD
        clf = LinearSVC(random_state=0)
        cv_scores = cross_val_score(clf, coefs_vec, classes, cv=cv)
        mean_scores = np.mean(cv_scores) * 100
        print measure, mean_scores
        std_scores = np.std(cv_scores) * 100
        plt.bar(bar_position, mean_scores, yerr=std_scores,
                color=color, align="center")
        bar_position += 1
    bar_position += 1

tick_position = len(measures) / 2 + np.array([0, len(measures) + 1])
plt.xticks(tick_position, ['site prediction', 'patient prediction'])
plt.ylabel('Classification accuracy (%)')
plt.legend(measures)
plt.show()
