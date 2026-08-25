"""
Functional connectivity predicts age group
==========================================

This example compares different kinds of :term:`functional connectivity`:
correlation, partial correlation, and tangent space embedding.
The resulting connectivity coefficients are then used to classify
participants by age group; i.e., children or adults.

This is a small example to showcase Nilearn features;
see :footcite:t:`Dadi2019` for a careful study.
"""

# %%
# Load brain development :term:`fMRI` dataset
# -------------------------------------------
# To save computation time, we will use only 60 subjects from
# the :func:`~nilearn.datasets.fetch_development_fmri` dataset.
from nilearn.datasets import fetch_atlas_msdl, fetch_development_fmri

development_dataset = fetch_development_fmri(n_subjects=60)

# %%
# Load Multi-Subject Dictionary Learning (MSDL) atlas and extract time series
# ---------------------------------------------------------------------------
# We use probabilistic regions of interest (ROIs) defined
# using the :func:`~nilearn.datasets.fetch_atlas_msdl` atlas.
#
# We then use the :class:`~nilearn.maskers.MultiNiftiMapsMasker` object
# to extract time series from the pre-defined ROIs for each subject.
# We could instead do this iteratively for each subject
# using :class:`~nilearn.maskers.NiftiMapsMasker` objects,
# but :class:`~nilearn.maskers.MultiNiftiMapsMasker` allows to
# extract time series for all subjects in a single step.

from nilearn.maskers import MultiNiftiMapsMasker

msdl_data = fetch_atlas_msdl()

masker = MultiNiftiMapsMasker(
    msdl_data.maps,
    resampling_target="data",
    t_r=development_dataset.t_r,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    memory="nilearn_cache",
    memory_level=1,
    standardize_confounds=True,
    standardize="zscore_sample",
    verbose=1,
)
masked_data = masker.fit_transform(
    development_dataset.func, confounds=development_dataset.confounds
)

# %%
# What kind of connectivity is most powerful for classification?
# --------------------------------------------------------------
# We will next define connectivity matrices using
# :func:`~nilearn.connectome.ConnectivityMeasure`
# so that we can use each matrix as features in our classification analysis.
# Since our goal is to compare different kinds of connectivity,
# we will define three different kinds of connectivity matrices,
# (i.e., "correlation", "partial correlation", "tangent")
# using the ``kind`` parameter.
#
# To do this, we will use a scikit-learn :class:`~sklearn.pipeline.Pipeline` to
# combine the connectivity measure and the classifier into a single object.
# Specifically, we will create a composite estimator with
# :func:`~nilearn.connectome.ConnectivityMeasure` and
# :class:`sklearn.model_selection.GridSearchCV` to perform a grid search
# over the ``C`` regularization parameter for our
# classifier, a :class:`sklearn.svm.LinearSVC`.

# prepare the classification pipeline
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nilearn.connectome import ConnectivityMeasure

kinds = ["correlation", "partial correlation", "tangent"]

pipe = Pipeline(
    [
        (
            "connectivity",
            ConnectivityMeasure(
                vectorize=True,
            ),
        ),
        (
            "classifier",
            GridSearchCV(
                LinearSVC(dual=True, random_state=0),
                {"C": [0.1, 1.0, 10.0]},
                cv=5,
            ),
        ),
    ]
)

param_grid = [
    {"classifier": [DummyClassifier(strategy="most_frequent")]},
    {"connectivity__kind": kinds},
]

# %%
# Rather than evaluating on the whole sample of 60 subjects,
# we want to cross-validate our model.
# We can directly pass the Pipeline object we defined above to
# a cross-validation estimator.
#
# Given the unbalanced sampling of age groups in
# :func:`~nilearn.datasets.fetch_development_fmri`,
# we will use stratified cross-validation to preserve the proportion of
# each age group in the training and testing sets.
# We will define 30 random,
# stratified splits of the subjects into training and testing sets,
# using :class:`sklearn.model_selection.StratifiedShuffleSplit`.

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

groups = development_dataset.phenotypic["Child_Adult"].to_list()
classes = LabelEncoder().fit_transform(groups)

cv = StratifiedShuffleSplit(n_splits=30, random_state=0, test_size=10)
gs = GridSearchCV(
    pipe,
    param_grid,
    scoring="accuracy",
    cv=cv,
    refit=False,
    n_jobs=2,
)
gs.fit(masked_data, classes)
mean_scores = gs.cv_results_["mean_test_score"]
scores_std = gs.cv_results_["std_test_score"]

# %%
# Finally, we display the classification accuracy results:
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 4), constrained_layout=True)

positions = [0.1, 0.2, 0.3, 0.4]
plt.barh(positions, mean_scores, align="center", height=0.05, xerr=scores_std)
yticks = ["dummy", *list(gs.cv_results_["param_connectivity__kind"].data[1:])]
yticks = [t.replace(" ", "\n") for t in yticks]
plt.yticks(positions, yticks)
plt.xlabel("Classification accuracy")
plt.gca().grid(True)
plt.gca().set_axisbelow(True)

# %%
# This is a small example to showcase Nilearn features.
# In practice, such comparisons need to be performed on
# much larger cohorts and replicated across several datasets.
# :footcite:t:`Dadi2019` showed
# that across many cohorts and clinical questions,
# tangent functional connectivity should be preferred.

plt.show()

# %%
# References
# ----------
#
# .. footbibliography::
