"""
SearchLight MVPA with a custom correlation estimator on Haxby data
===================================================================

This tutorial example shows how to use :class:`nilearn.decoding.SearchLight`
with a user-defined scikit-learn compatible estimator.

The workflow is:

1. Load the Haxby dataset and keep only ``face`` and ``house`` trials.
2. Define a custom ``CorrelationMVPA`` estimator that computes the
   Haxby-style correlation contrast
   ``(within-category similarity - between-category similarity) / 2``
   across run splits.
3. Run a whole-brain SearchLight with this estimator.
4. Visualize the resulting score map.
"""

# %%
# Load Haxby dataset
# ------------------
import numpy as np
import pandas as pd

from nilearn.datasets import fetch_haxby
from nilearn.image import load_img

# We fetch 2nd subject from haxby datasets (which is default)
haxby_dataset = fetch_haxby()

# print basic information on the dataset
print(f"Anatomical nifti image (3D) is located at: {haxby_dataset.mask}")
print(f"Functional nifti image (4D) is located at: {haxby_dataset.func[0]}")

fmri_filename = haxby_dataset.func[0]

labels = pd.read_csv(haxby_dataset.session_target[0], sep=" ")
y = labels["labels"]
run = labels["chunks"]

# %%
# Define our own MVPA estimator for use in SearchLight
from sklearn.base import BaseEstimator


def fisher_z(r, eps=1e-12):
    """Apply Fisher z-transform to correlation coefficient r."""
    # clip to avoid inf at ±1
    r = np.clip(r, -1 + eps, 1 - eps)
    return np.arctanh(r)


def pattern_corr(a, b):
    """Compute correlation between two patterns, with mean-centering and norm.

    This is a more stable and efficient way to compute correlation than
    np.corrcoef for 1D patterns.
    """
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


class CorrelationMVPA(BaseEstimator):
    """Haxby-style correlation MVPA score for a pair of labels.

    Computes (within - between)/2 using run splits provided in `groups`.

    Parameters
    ----------
    labels : tuple of str, default=("face", "house")
        The two condition labels to contrast. Must be present in `y`.
    split : str, default="parity"
        How to split runs for cross-validation.
        Only "parity" (even vs odd runs) is implemented here.
    fisher_z : bool, default=True
        Whether to apply Fisher z-transform to correlations before
        computing the final score.
    """

    nilearn_searchlight_uses_cv = False

    def __init__(
        self, labels=("face", "house"), split="parity", fisher_z=True
    ):
        self.labels = labels
        self.split = split
        self.fisher_z_ = fisher_z

    def fit(self, X, y, groups=None):
        """Fit the estimator and store a single correlation-based score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data matrix for one SearchLight sphere. Rows are samples
            (volumes) and columns are voxel features in the sphere.

        y : array-like of shape (n_samples,)
            Condition labels for each sample. Must contain both labels
            specified in ``self.labels``.

        groups : array-like of shape (n_samples,), default=None
            Run/chunk assignment per sample. Required to create two splits
            (currently parity-based: even runs vs odd runs).

        Returns
        -------
        self : CorrelationMVPA
            Fitted estimator with ``score_`` set to the MVPA contrast.
            If any required condition/split combination is missing,
            ``score_`` is set to ``NaN``.
        """
        if groups is None:
            raise ValueError(
                "groups (runs/chunks) are required for CorrelationMVPA."
            )

        a, b = self.labels
        y = np.asarray(y)
        groups = np.asarray(groups)

        if self.split == "parity":
            g1 = groups % 2 == 0
            g2 = ~g1
        else:
            raise ValueError("Only split='parity' implemented here.")

        def mean_pattern(lbl, mask):
            sel = (y == lbl) & mask
            if not np.any(sel):
                return None
            return X[sel].mean(axis=0)

        a1 = mean_pattern(a, g1)
        a2 = mean_pattern(a, g2)
        b1 = mean_pattern(b, g1)
        b2 = mean_pattern(b, g2)
        if any(v is None for v in (a1, a2, b1, b2)):
            self.score_ = float("nan")
            return self

        r_aa = pattern_corr(a1, a2)
        r_bb = pattern_corr(b1, b2)
        r_ab = pattern_corr(a1, b2)
        r_ba = pattern_corr(b1, a2)

        if self.fisher_z_:
            r_aa, r_bb, r_ab, r_ba = map(fisher_z, (r_aa, r_bb, r_ab, r_ba))

        self.score_ = 0.5 * ((r_aa + r_bb) - (r_ab + r_ba))
        return self

    def score(self, X, y=None, groups=None):
        """
        Return the score computed during fitting.

        Parameters
        ----------
        X, y, groups : ignored
             These parameters are required by the sklearn API but are not used
             here since the score is pre-computed in fit.
        """
        # SearchLight can call this after fit
        del X, y, groups  # unused, required by sklearn API
        return self.score_


# %%
# Restrict to faces and houses
# ----------------------------
from nilearn.image import index_img, mean_img

condition_mask = y.isin(["face", "house"])

fmri_img = index_img(fmri_filename, condition_mask)
y, run = y[condition_mask], run[condition_mask]

# Overview of the input data

n_labels = len(np.unique(y))

print(f"{n_labels} labels (y): {np.unique(y)}")
print(f"fMRI data shape (X): {fmri_img.shape}")
print(f"Runs (groups): {np.unique(run)}")

# %%
# Perform searchlight analysis, using the CorrelationMVPA estimator
# defined above
from nilearn.decoding import SearchLight

mask_img = load_img(haxby_dataset.mask)

searchlight = SearchLight(
    mask_img=mask_img,
    process_mask_img=None,
    radius=5.6,
    n_jobs=2,
    verbose=1,
    estimator=CorrelationMVPA(
        labels=("face", "house"), split="parity", fisher_z=True
    ),
)
searchlight.fit(imgs=fmri_img, y=y, groups=run)
scores_img = searchlight.scores_img_

# %%
# Visualize the searchlight scores
from nilearn.plotting import plot_img

mean_fmri = mean_img(fmri_img)

plot_img(
    scores_img,
    bg_img=mean_fmri,
    title="Searchlight scores (face vs house)",
    display_mode="z",
    cut_coords=[-9],
    vmin=0.0,
    cmap="inferno",
    black_bg=True,
    colorbar=True,
)

# %%
