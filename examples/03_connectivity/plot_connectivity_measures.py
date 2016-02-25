"""
Comparing different functional connectivity measures
====================================================

This example compares different measures of functional connectivity between
regions of interest : correlation, partial correlation, as well as a measure
called tangent. The resulting connectivity coefficients are used to
classify ADHD vs control subjects and the tangent measure outperforms the
standard measures.

"""

# Fetch dataset
import nilearn.datasets
atlas = nilearn.datasets.fetch_atlas_msdl()
dataset = nilearn.datasets.fetch_adhd(n_subjects=30)


######################################################################
# Extract regions time series signals
import nilearn.input_data
masker = nilearn.input_data.NiftiMapsMasker(
    atlas.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=None, t_r=2.5, standardize=False,
    memory='nilearn_cache', memory_level=1)
subjects = []
sites = []
adhds = []
for func_file, phenotypic in zip(dataset.func, dataset.phenotypic):
    # keep only 3 sites, to save computation time
    if phenotypic['site'] in [b'"NYU"', b'"OHSU"', b'"NeuroImage"']:
        time_series = masker.fit_transform(func_file)
        subjects.append(time_series)
        sites.append(phenotypic['site'])
        adhds.append(phenotypic['adhd'])  # ADHD/control label


######################################################################
# Estimate connectivity
import nilearn.connectome
kinds = ['tangent', 'partial correlation', 'correlation']
individual_connectivity_matrices = {}
mean_connectivity_matrix = {}
for kind in kinds:
    conn_measure = nilearn.connectome.ConnectivityMeasure(kind=kind)
    individual_connectivity_matrices[kind] = conn_measure.fit_transform(
        subjects)
    # Compute the mean connectivity
    if kind == 'tangent':
        mean_connectivity_matrix[kind] = conn_measure.mean_
    else:
        mean_connectivity_matrix[kind] = \
            individual_connectivity_matrices[kind].mean(axis=0)


######################################################################
# Plot the mean connectome
import numpy as np
import nilearn.plotting
labels = atlas.labels
region_coords = atlas.region_coords
for kind in kinds:
    nilearn.plotting.plot_connectome(mean_connectivity_matrix[kind],
                                     region_coords, edge_threshold='98%',
                                     title=kind)


######################################################################
# Use the connectivity coefficients to classify ADHD vs controls
from sklearn.svm import LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
classes = ['{0}{1}'.format(site, adhd) for site, adhd in zip(sites, adhds)]
print('Classification accuracy:')
mean_scores = []
cv = StratifiedKFold(classes, n_folds=3)
for kind in kinds:
    svc = LinearSVC()
    # Transform the connectivity matrices to 1D arrays
    coonectivity_coefs = nilearn.connectome.sym_to_vec(
        individual_connectivity_matrices[kind])
    cv_scores = cross_val_score(svc, coonectivity_coefs,
                                adhds, cv=cv, scoring='accuracy')
    print('%20s score: %1.2f +- %1.2f' % (kind, cv_scores.mean(),
                                          cv_scores.std()))
    mean_scores.append(cv_scores.mean())


######################################################################
# Display the classification scores
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()
