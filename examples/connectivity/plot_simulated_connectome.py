"""
Connectivity structure estimation on simulated data
===================================================

This example shows a comparison of graph lasso and group-sparse covariance
estimation of connectivity structre for a synthetic dataset.

"""

import matplotlib.pyplot as plt


def plot_matrix(m, ylabel=""):
    abs_max = abs(m).max()
    plt.imshow(m, cmap=plt.cm.RdBu_r, interpolation="nearest",
              vmin=-abs_max, vmax=abs_max)


# Generate synthetic data
from nilearn._utils.testing import generate_group_sparse_gaussian_graphs

n_subjects = 20  # number of subjects
n_displayed = 3  # number of subjects displayed
subjects, precisions, topology = generate_group_sparse_gaussian_graphs(
    n_subjects=n_subjects, n_features=10, min_n_samples=30, max_n_samples=50,
    density=0.1)

fig = plt.figure(figsize=(10, 7))
plt.subplots_adjust(hspace=0.4)
for n in range(n_displayed):
    plt.subplot(n_displayed, 4, 4 * n + 1)
    plot_matrix(precisions[n])
    if n == 0:
        plt.title("ground truth")
    plt.ylabel("subject %d" % n)


# Run group-sparse covariance on all subjects
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(max_iter=50, verbose=1)
gsc.fit(subjects)

for n in range(n_displayed):
    plt.subplot(n_displayed, 4, 4 * n + 2)
    plot_matrix(gsc.precisions_[..., n])
    if n == 0:
        plt.title("group-sparse\n$\\alpha=%.2f$" % gsc.alpha_)


# Fit one graph lasso per subject
from sklearn.covariance import GraphLassoCV
gl = GraphLassoCV(verbose=1)

for n, subject in enumerate(subjects[:n_displayed]):
    gl.fit(subject)

    plt.subplot(n_displayed, 4, 4 * n + 3)
    plot_matrix(gl.precision_)
    if n == 0:
        plt.title("graph lasso")
    plt.ylabel("$\\alpha=%.2f$" % gl.alpha_)


# Fit one graph lasso for all subjects at once
import numpy as np
gl.fit(np.concatenate(subjects))

plt.subplot(n_displayed, 4, 4)
plot_matrix(gl.precision_)
plt.title("graph lasso, all subjects\n$\\alpha=%.2f$" % gl.alpha_)

plt.show()
