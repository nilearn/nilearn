"""
Group Sparse inverse covariance for multi-subject connectome
============================================================

This example shows how to estimate a connectome on a group of subjects
using the group sparse inverse covariance estimate.

This example is a toy example as running it on more subjects will
require a longer run time.

"""

# %%
import numpy as np

from nilearn.plotting import plot_matrix


def plot_matrices(cov, prec, title, labels):
    """Plot covariance and precision matrices, for a given processing."""
    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plot_matrix(
        cov,
        vmin=-1,
        vmax=1,
        title=f"{title} / covariance",
        labels=labels,
    )
    # Display precision matrix
    plot_matrix(
        prec,
        vmin=-span,
        vmax=span,
        title=f"{title} / precision",
        labels=labels,
    )


# %%
# Fetching datasets
# ------------------
from nilearn.datasets import fetch_atlas_msdl, fetch_development_fmri

n_subjects = 4  # subjects to consider for group-sparse covariance (max: 40)


msdl_atlas_dataset = fetch_atlas_msdl()
rest_dataset = fetch_development_fmri(n_subjects=n_subjects)

# print basic information on the dataset
print(
    f"First subject functional nifti image (4D) is at: {rest_dataset.func[0]}"
)


# %%
# Extracting region signals
# -------------------------
from nilearn.maskers import MultiNiftiMapsMasker

masker = MultiNiftiMapsMasker(
    msdl_atlas_dataset.maps,
    resampling_target="maps",
    detrend=True,
    high_variance_confounds=True,
    low_pass=None,
    high_pass=0.01,
    t_r=rest_dataset.t_r,
    standardize="zscore_sample",
    standardize_confounds=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=1,
)

func_filenames = rest_dataset.func
confound_filenames = rest_dataset.confounds

subject_time_series = masker.fit_transform(
    func_filenames, confounds=confound_filenames
)


# %%
# Computing group-sparse precision matrices
# -----------------------------------------
from nilearn.connectome import GroupSparseCovarianceCV

gsc = GroupSparseCovarianceCV(verbose=1)
gsc.fit(subject_time_series)

# %%
from sklearn.covariance import GraphicalLassoCV

gl = GraphicalLassoCV(verbose=True)
gl.fit(np.concatenate(subject_time_series))


# %%
# Displaying results
# ------------------
from nilearn.plotting import (
    find_probabilistic_atlas_cut_coords,
    plot_connectome,
    show,
)

atlas_img = msdl_atlas_dataset.maps
atlas_region_coords = find_probabilistic_atlas_cut_coords(atlas_img)
labels = msdl_atlas_dataset.labels

plot_connectome(
    gl.covariance_,
    atlas_region_coords,
    edge_threshold="90%",
    title="Covariance",
    display_mode="lzr",
)

#  %%
plot_connectome(
    -gl.precision_,
    atlas_region_coords,
    edge_threshold="90%",
    title="Sparse inverse covariance (GraphicalLasso)",
    display_mode="lzr",
    edge_vmax=0.5,
    edge_vmin=-0.5,
)
plot_connectome(
    -gsc.precisions_[..., 0],
    atlas_region_coords,
    edge_threshold="90%",
    title="GroupSparseCovariance",
    display_mode="lzr",
    edge_vmax=0.5,
    edge_vmin=-0.5,
)

show()

#  %%
plot_matrices(gl.covariance_, gl.precision_, "GraphicalLasso", labels)
plot_matrices(
    gsc.covariances_[..., 0],
    gsc.precisions_[..., 0],
    "GroupSparseCovariance",
    labels,
)

show()
