"""
Computation of covariance matrix between brain regions
======================================================

This example shows how to extract fMRI signals from atlas-derived regions
and to estimate covariance matrices from these.
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory

import nilearn
from nilearn import plotting, image
from nilearn.plotting import cm

n_subjects = 3  # Number of subjects to consider for group-sparse covariance
subject_to_plot = 0  # subject to plot
n_jobs = 1  # number of processes to work in parallel


def plot_connectome(cov, atlas_maps, **kwargs):
    """Plot connectome given a covariance matrix and atlas maps"""
    imgs = image.iter_img(atlas_maps)
    regions_coords = np.array(
        [map(np.asscalar,
         np.array(plotting.find_xyz_cut_coords(img))) for img in imgs])
    np.random.seed(42)
    node_colors = np.random.rand(len(regions_coords) // 2, 3)
    node_colors = np.concatenate([node_colors, node_colors], axis=0)
    node_colors = np.sort(node_colors, axis=0)
    plotting.plot_connectome(cov, regions_coords,
                             node_size=50, node_color=node_colors,
                             **kwargs)


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    # Compute sparsity pattern
    sparsity = prec == 0

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest", vmin=-1, vmax=1, cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)

    # Display sparsity pattern
    plt.figure()
    plt.imshow(sparsity, interpolation="nearest")
    plt.title("%s / sparsity" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest", vmin=-span, vmax=span,
               cmap=cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)


# Fetching datasets ###########################################################
print("-- Retrieving atlas and ADHD resting-state data ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# Extracting region signals ###################################################
import nibabel
import nilearn.image
import nilearn.input_data

mem = Memory('nilearn_cache')  # setup persistence framework

maps_img = nibabel.load(msdl_atlas_dataset.maps)

masker = nilearn.input_data.NiftiMapsMasker(
    maps_img, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

subjects = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for func_filename, confound_filename in zip(func_filenames,
                                            confound_filenames):
    print("Processing file %s" % func_filename)

    print("-- Computing confounds ...")
    hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
        func_filename)

    print("-- Computing region signals ...")
    region_ts = masker.transform(func_filename,
                                 confounds=[hv_confounds, confound_filename])
    subjects.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(n_jobs=n_jobs, verbose=2)
gsc.fit(subjects)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(n_jobs=n_jobs, verbose=2)
gl.fit(subjects[subject_to_plot])

# Displaying results ##########################################################
print("-- Displaying results")
title = "Subject {0:d}: GroupSparseCovariance $\\alpha={1:.2e}$".format(
    subject_to_plot + 1,
    gsc.alpha_)

plot_connectome(gsc.covariances_[..., subject_to_plot],
                maps_img, edge_threshold='80%',
                title=title)
plot_matrices(gsc.covariances_[..., subject_to_plot],
              gsc.precisions_[..., subject_to_plot], title)

title = "Subject {0:d}: GraphLasso $\\alpha={1:.2e}$".format(
    subject_to_plot + 1,
    gl.alpha_)
plot_connectome(gl.covariance_,
                maps_img, edge_threshold='80%',
                title=title)
plot_matrices(gl.covariance_, gl.precision_, title)

plt.show()

imgs = list(image.iter_img(maps_img))
coords = np.array([
    map(np.asscalar,
        np.array(plotting.find_xyz_cut_coords(img))) for img in imgs])

cov = gsc.covariances_[..., subject_to_plot]

from nilearn import plotting
np.random.seed(42)
node_colors = np.random.rand(maps_img.shape[-1], 3)
display = plotting.plot_connectome(cov, coords,
                                   edge_threshold=0.38,
                                   title='threshold=0.38',
                                   node_size=50, node_color=node_colors)

display = plotting.plot_connectome(cov, coords,
                                   edge_threshold='70%',
                                   title='threshold=70%',
                                   node_size=50, node_color=node_colors)
