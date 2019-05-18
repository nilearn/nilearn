"""
Functional connectivity predicts age group
==========================================

This example compares different kinds of functional connectivity between
regions of interest : correlation, partial correlation, and tangent space
embedding.

The resulting connectivity coefficients can be used to
discriminate children from adults. In general, the **tangent kind**
**outperforms** the standard correlations: see `Dadi et al 2019
<https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
for a careful study.
"""
# Matrix plotting from Nilearn: nilearn.plotting.plot_matrix
import numpy as np
import matplotlib.pylab as plt


def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)


###############################################################################
# Load brain development fMRI dataset and MSDL atlas
# -------------------------------------------------------------------
# We study only 30 subjects from the dataset, to save computation time.
from nilearn import datasets

rest_data = datasets.fetch_development_fmri(n_subjects=30)

###############################################################################
# We use probabilistic regions of interest (ROIs) from the MSDL atlas.
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
    n_regions, msdl_data.networks))

###############################################################################
# Region signals extraction
# -------------------------
# To extract regions time series, we instantiate a
# :class:`nilearn.input_data.NiftiMapsMasker` object and pass the atlas the
# file name to it, as well as filtering band-width and detrending option.
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)

###############################################################################
# Then we compute region signals and extract useful phenotypic informations.
children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
        rest_data.func, rest_data.confounds, rest_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_child = phenotypic['Child_Adult'] == 'child'
    if is_child:
        children.append(time_series)

    groups.append(phenotypic['Child_Adult'])

print('Data has {0} children.'.format(len(children)))

###############################################################################
# ROI-to-ROI correlations of children
# -----------------------------------
# The simpler and most commonly used kind of connectivity is correlation. It
# models the full (marginal) connectivity between pairwise ROIs. We can
# estimate it using :class:`nilearn.connectome.ConnectivityMeasure`.
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

###############################################################################
# From the list of ROIs time-series for children, the
# `correlation_measure` computes individual correlation matrices.
correlation_matrices = correlation_measure.fit_transform(children)

# All individual coefficients are stacked in a unique 2D matrix.
print('Correlations of children are stacked in an array of shape {0}'
      .format(correlation_matrices.shape))

###############################################################################
# as well as the average correlation across all fitted subjects.
mean_correlation_matrix = correlation_measure.mean_
print('Mean correlation has shape {0}.'.format(mean_correlation_matrix.shape))

###############################################################################
# We display the connectome matrices of the first 4 children
from nilearn import plotting

plot_matrices(correlation_matrices[:4], 'correlation')
###############################################################################
# The blocks structure that reflect functional networks are visible.

###############################################################################
# Now we display as a connectome the mean correlation matrix over all children.
plotting.plot_connectome(mean_correlation_matrix, msdl_coords,
                         title='mean correlation over all children')

###############################################################################
# Studying partial correlations
# -----------------------------
# We can also study **direct connections**, revealed by partial correlation
# coefficients. We just change the `ConnectivityMeasure` kind
partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')

###############################################################################
# and repeat the previous operation.
partial_correlation_matrices = partial_correlation_measure.fit_transform(
    children)

###############################################################################
# Most of direct connections are weaker than full connections,
plot_matrices(partial_correlation_matrices[:4], 'partial')

###############################################################################
# Compared to a connectome computed on correlations, the connectome graph
# with partial correlations is more sparse:
plotting.plot_connectome(
    partial_correlation_measure.mean_, msdl_coords,
    title='mean partial correlation over all children')

###############################################################################
# Extract subjects variabilities around a group connectivity
# ----------------------------------------------------------
# We can use **both** correlations and partial correlations to capture
# reproducible connectivity patterns at the group-level and build a **robust**
# **group connectivity matrix**. This is done by the **tangent** kind.
tangent_measure = ConnectivityMeasure(kind='tangent')

###############################################################################
# We fit our children group and get the group connectivity matrix stored as
# in `tangent_measure.mean_`, and individual deviation matrices of each subject
# from it.
tangent_matrices = tangent_measure.fit_transform(children)

###############################################################################
# `tangent_matrices` model individual connectivities as
# **perturbations** of the group connectivity matrix `tangent_measure.mean_`.
# Keep in mind that these subjects-to-group variability matrices do not
# straight reflect individual brain connections. For instance negative
# coefficients can not be interpreted as anticorrelated regions.
plot_matrices(tangent_matrices[:4], 'tangent variability')

###############################################################################
# The average tangent matrix cannot be interpreted, as the average
# variation is expected to be zero

###############################################################################
# What kind of connectivity is most powerful for classification?
# --------------------------------------------------------------
# we will use connectivity matrices as features to distinguish children from
# adults. We use cross-validation and measure classification accuracy to
# compare the different kinds of connectivity matrices.

# prepare the classification pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

kinds = ['correlation', 'partial correlation', 'tangent']

# *ConnectivityMeasure* can output the estimated subjects coefficients
# as a 1D arrays through the parameter *vectorize*.
pipe = Pipeline(
    [('connectivity', ConnectivityMeasure(vectorize=True)),
     ('classifier', LinearSVC())])

param_grid = [
    {'classifier': [DummyClassifier('most_frequent')]},
    {'connectivity__kind': kinds}
]


######################################################################
# We use random splits of the subjects into training/testing sets.
# StratifiedShuffleSplit allows preserving the proportion of children in the
# test set.
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

_, classes = np.unique(groups, return_inverse=True)
cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
gs = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=cv, verbose=1,
                  refit=False)
gs.fit(pooled_subjects, classes)
mean_scores = gs.cv_results_['mean_test_score']
scores_std = gs.cv_results_['std_test_score']

######################################################################
# display the results
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds) + 1) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05, xerr=scores_std)
yticks = ['dummy'] + list(gs.cv_results_['param_connectivity__kind'].data[1:])
yticks = [t.replace(' ', '\n') for t in yticks]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.tight_layout()


###############################################################################
# This is a small example to showcase nilearn features. In practice such
# comparisons need to be performed on much larger cohorts and several
# datasets.
# `Dadi et al 2019
# <https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
# Showed that across many cohorts and clinical questions, the tangent
# kind should be preferred.

plotting.show()
