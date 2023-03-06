"""
Connectivity structure estimation on simulated data
===================================================

This example shows a comparison of graph lasso and group-sparse covariance
estimation of connectivity structure for a synthetic dataset.

"""

import matplotlib.pyplot as plt

# Generate synthetic data
from nilearn._utils.data_gen import generate_group_sparse_gaussian_graphs
from nilearn.plotting import show

n_subjects = 20
n_displayed = 3
subjects, precisions, _ = generate_group_sparse_gaussian_graphs(
    n_subjects=n_subjects,
    n_features=10,
    min_n_samples=30,
    max_n_samples=50,
    density=0.1,
)

from nilearn import plotting

fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(hspace=0.4)
for n in range(n_displayed):
    ax = plt.subplot(n_displayed, 4, 4 * n + 1)
    max_precision = precisions[n].max()
    plotting.plot_matrix(
        precisions[n],
        vmin=-max_precision,
        vmax=max_precision,
        axes=ax,
        colorbar=False,
    )

    if n == 0:
        plt.title("ground truth")
    plt.ylabel(f"subject {int(n)}")


# Run group-sparse covariance on all subjects
from nilearn.connectome import GroupSparseCovarianceCV

gsc = GroupSparseCovarianceCV(max_iter=50, verbose=1)
gsc.fit(subjects)

for n in range(n_displayed):
    ax = plt.subplot(n_displayed, 4, 4 * n + 2)
    max_precision = gsc.precisions_[..., n].max()
    plotting.plot_matrix(
        gsc.precisions_[..., n],
        axes=ax,
        vmin=-max_precision,
        vmax=max_precision,
        colorbar=False,
    )
    if n == 0:
        plt.title(f"group-sparse\n$\\alpha={gsc.alpha_:.2f}$")


# Fit one graph lasso per subject
try:
    from sklearn.covariance import GraphicalLassoCV
except ImportError:
    # for Scitkit-Learn < v0.20.0
    from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

gl = GraphicalLassoCV(verbose=1)

for n, subject in enumerate(subjects[:n_displayed]):
    gl.fit(subject)

    ax = plt.subplot(n_displayed, 4, 4 * n + 3)
    max_precision = gl.precision_.max()
    plotting.plot_matrix(
        gl.precision_,
        axes=ax,
        vmin=-max_precision,
        vmax=max_precision,
        colorbar=False,
    )
    if n == 0:
        plt.title("graph lasso")
    plt.ylabel(f"$\\alpha={gl.alpha_:.2f}$")


# Fit one graph lasso for all subjects at once
import numpy as np

gl.fit(np.concatenate(subjects))

ax = plt.subplot(n_displayed, 4, 4)
max_precision = gl.precision_.max()
plotting.plot_matrix(
    gl.precision_,
    axes=ax,
    vmin=-max_precision,
    vmax=max_precision,
    colorbar=False,
)
plt.title(f"graph lasso, all subjects\n$\\alpha={gl.alpha_:.2f}$")

show()
