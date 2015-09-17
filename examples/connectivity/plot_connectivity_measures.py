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

# Estimate connectivity
import nilearn.connectivity
kinds = ['robust dispersion', 'partial correlation', 'precision',
         'correlation', 'covariance']
individual_connectivity_matrices = {}
mean_connectivity_matrix = {}
for kind in kinds:
    conn_measure = nilearn.connectivity.ConnectivityMeasure(kind=kind)
    individual_connectivity_matrices[kind] = conn_measure.fit_transform(
        subjects)
    # Compute the mean connectivity
    if kind == 'robust dispersion':
        mean_connectivity_matrix[kind] = conn_measure.robust_mean_
    else:
        mean_connectivity_matrix[kind] = \
            individual_connectivity_matrices[kind].mean(axis=0)

# Plot the correlation matrix for one subject
subject_n = 28
import matplotlib.pyplot as plt
plt.figure()
plt.imshow(individual_connectivity_matrices['correlation'][subject_n],
           interpolation="nearest", vmin=-1., vmax=1.)
plt.title('subject %d, correlation' % subject_n)

# Plot the mean connectome
import numpy as np
import nilearn.plotting
labels = np.recfromcsv(atlas.labels)
region_coords = labels[['x', 'y', 'z']].tolist()
for kind in ['robust dispersion', 'correlation', 'partial correlation']:
    nilearn.plotting.plot_connectome(mean_connectivity_matrix[kind],
                                     region_coords, edge_threshold='98%',
                                     title=kind)

# Get site and ADHD/control label for each subject
adhds = dataset.phenotypic['adhd']
sites = ['"Peking"' if 'Peking' in site else site for site in
         dataset.phenotypic['site']]  # Group Peking sites

# Use the connectivity coefficients to classify ADHD vs controls
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
classes = [site + str(adhd) for site, adhd in zip(sites, adhds)]
print('Classification accuracy:')
mean_scores = []
cv = StratifiedKFold(classes, n_folds=3)
for kind in kinds:
    svc = LinearSVC()
    # Transform the connectivity matrices to 1D arrays
    coonectivity_coefs = nilearn.connectivity.sym_to_vec(
        individual_connectivity_matrices[kind])
    cv_scores = cross_val_score(svc, coonectivity_coefs,
                                adhds, cv=cv, scoring='accuracy')
    print('%20s score: %1.2f +- %1.2f' % (kind, cv_scores.mean(),
                                          cv_scores.std()))
    mean_scores.append(cv_scores.mean())

# Display the classification scores
plt.figure()
positions = np.arange(len(kinds)) * .5 + .5
plt.barh(positions, mean_scores, align='center', height=0.2)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.show()
