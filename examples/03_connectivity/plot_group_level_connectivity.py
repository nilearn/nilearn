"""
Classification of age groups using functional connectivity
==========================================================

This example compares different kinds of functional connectivity between
regions of interest : correlation, partial correlation, and tangent space
embedding.

The resulting connectivity coefficients can be used to
discriminate children from adults. In general, the tangent space embedding
**outperforms** the standard correlations: see `Dadi et al 2019
<https://www.sciencedirect.com/science/article/pii/S1053811919301594>`_
for a careful study.

.. include:: ../../../examples/masker_note.rst

"""

###############################################################################
# Load brain development fMRI dataset and MSDL atlas
# --------------------------------------------------
# We study only 30 subjects from the dataset, to save computation time.
from nilearn import datasets

development_dataset = datasets.fetch_development_fmri(n_subjects=30)

###############################################################################
# We use probabilistic regions of interest (ROIs) from the MSDL atlas.
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print(
    "MSDL has {0} ROIs, part of the following networks :\n{1}.".format(
        n_regions, msdl_data.networks
    )
)

###############################################################################
# Region signals extraction
# -------------------------
# To extract regions time series, we instantiate a
# :class:`nilearn.maskers.NiftiMapsMasker` object and pass the atlas the
# file name to it, as well as filtering band-width and detrending option.
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    msdl_data.maps,
    resampling_target="data",
    t_r=2,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    memory="nilearn_cache",
    memory_level=1,
).fit()

###############################################################################
# Then we compute region signals and extract useful phenotypic information.
children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
    development_dataset.func,
    development_dataset.confounds,
    development_dataset.phenotypic,
):
    time_series = masker.transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    if phenotypic["Child_Adult"] == "child":
        children.append(time_series)
    groups.append(phenotypic["Child_Adult"])

print(f"Data has {len(children)} children.")

###############################################################################
# ROI-to-ROI correlations of children
# -----------------------------------
# The simpler and most commonly used kind of connectivity is correlation. It
# models the full (marginal) connectivity between pairwise ROIs. We can
# estimate it using :class:`nilearn.connectome.ConnectivityMeasure`.
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind="correlation")

###############################################################################
# From the list of ROIs time-series for children, the
# `correlation_measure` computes individual correlation matrices.
correlation_matrices = correlation_measure.fit_transform(children)

# All individual coefficients are stacked in a unique 2D matrix.
print(
    "Correlations of children are stacked in an array of shape {0}".format(
        correlation_matrices.shape
    )
)

###############################################################################
# as well as the average correlation across all fitted subjects.
mean_correlation_matrix = correlation_measure.mean_
print(f"Mean correlation has shape {mean_correlation_matrix.shape}.")

from matplotlib import pyplot as plt

###############################################################################
# We display the connectome matrices of the first 3 children
from nilearn import plotting

_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(correlation_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",
        colorbar=False,
        axes=ax,
        title=f"correlation, child {i}",
    )
###############################################################################
# The blocks structure that reflect functional networks are visible.

###############################################################################
# Now we display as a connectome the mean correlation matrix over all children.
plotting.plot_connectome(
    mean_correlation_matrix,
    msdl_coords,
    title="mean correlation over all children",
)

###############################################################################
# Studying partial correlations
# -----------------------------
# We can also study **direct connections**, revealed by partial correlation
# coefficients. We just change the `ConnectivityMeasure` kind
partial_correlation_measure = ConnectivityMeasure(kind="partial correlation")
partial_correlation_matrices = partial_correlation_measure.fit_transform(
    children
)

###############################################################################
# Most of direct connections are weaker than full connections.

_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(partial_correlation_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",
        colorbar=False,
        axes=ax,
        title=f"partial correlation, child {i}",
    )
###############################################################################
plotting.plot_connectome(
    partial_correlation_measure.mean_,
    msdl_coords,
    title="mean partial correlation over all children",
)

###############################################################################
# Extract subjects variabilities around a group connectivity
# ----------------------------------------------------------
# We can use **both** correlations and partial correlations to capture
# reproducible connectivity patterns at the group-level.
# This is done by the tangent space embedding.
tangent_measure = ConnectivityMeasure(kind="tangent")

###############################################################################
# We fit our children group and get the group connectivity matrix stored as
# in `tangent_measure.mean_`, and individual deviation matrices of each subject
# from it.
tangent_matrices = tangent_measure.fit_transform(children)

###############################################################################
# `tangent_matrices` model individual connectivities as
# **perturbations** of the group connectivity matrix `tangent_measure.mean_`.
# Keep in mind that these subjects-to-group variability matrices do not
# directly reflect individual brain connections. For instance negative
# coefficients can not be interpreted as anticorrelated regions.
_, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (matrix, ax) in enumerate(zip(tangent_matrices, axes)):
    plotting.plot_matrix(
        matrix,
        tri="lower",
        colorbar=False,
        axes=ax,
        title=f"tangent offset, child {i}",
    )


###############################################################################
# The average tangent matrix cannot be interpreted, as individual matrices
# represent deviations from the mean, which is set to 0.

###############################################################################
# What kind of connectivity is most powerful for classification?
# --------------------------------------------------------------
# We will use connectivity matrices as features to distinguish children from
# adults. We use cross-validation and measure classification accuracy to
# compare the different kinds of connectivity matrices.
# We use random splits of the subjects into training/testing sets.
# StratifiedShuffleSplit allows preserving the proportion of children in the
# test set.
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC

kinds = ["correlation", "partial correlation", "tangent"]
_, classes = np.unique(groups, return_inverse=True)
cv = StratifiedShuffleSplit(n_splits=15, random_state=0, test_size=5)
pooled_subjects = np.asarray(pooled_subjects)

scores = {}
for kind in kinds:
    scores[kind] = []
    for train, test in cv.split(pooled_subjects, classes):
        # *ConnectivityMeasure* can output the estimated subjects coefficients
        # as a 1D arrays through the parameter *vectorize*.
        connectivity = ConnectivityMeasure(kind=kind, vectorize=True)
        # build vectorized connectomes for subjects in the train set
        connectomes = connectivity.fit_transform(pooled_subjects[train])
        # fit the classifier
        classifier = LinearSVC().fit(connectomes, classes[train])
        # make predictions for the left-out test subjects
        predictions = classifier.predict(
            connectivity.transform(pooled_subjects[test])
        )
        # store the accuracy for this cross-validation fold
        scores[kind].append(accuracy_score(classes[test], predictions))


######################################################################
# display the results

mean_scores = [np.mean(scores[kind]) for kind in kinds]
scores_std = [np.std(scores[kind]) for kind in kinds]

plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * 0.1 + 0.1
plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
yticks = [k.replace(" ", "\n") for k in kinds]
plt.yticks(positions, yticks)
plt.gca().grid(True)
plt.gca().set_axisbelow(True)
plt.gca().axvline(0.8, color="red", linestyle="--")
plt.xlabel("Classification accuracy\n(red line = chance level)")
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
