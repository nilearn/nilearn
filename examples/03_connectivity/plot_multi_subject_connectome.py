"""
Group Sparse inverse covariance for multi-subject connectome
============================================================

This example shows how to estimate a connectome on a group of subjects
using the group sparse inverse covariance estimate.

.. include:: ../../../examples/masker_note.rst

"""
import numpy as np
from nilearn import plotting

n_subjects = 4  # subjects to consider for group-sparse covariance (max: 40)


def plot_matrices(cov, prec, title, labels):
    """Plot covariance and precision matrices, for a given processing."""
    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plotting.plot_matrix(
        cov,
        cmap=plotting.cm.bwr,
        vmin=-1,
        vmax=1,
        title=f"{title} / covariance",
        labels=labels,
    )
    # Display precision matrix
    plotting.plot_matrix(
        prec,
        cmap=plotting.cm.bwr,
        vmin=-span,
        vmax=span,
        title=f"{title} / precision",
        labels=labels,
    )


##############################################################################
# Fetching datasets
# ------------------
from nilearn import datasets

msdl_atlas_dataset = datasets.fetch_atlas_msdl()
rest_dataset = datasets.fetch_development_fmri(n_subjects=n_subjects)

# print basic information on the dataset
print(
    f"First subject functional nifti image (4D) is at: {rest_dataset.func[0]}"
)


##############################################################################
# Extracting region signals
# -------------------------
from nilearn.maskers import NiftiMapsMasker

masker = NiftiMapsMasker(
    msdl_atlas_dataset.maps,
    resampling_target="maps",
    detrend=True,
    high_variance_confounds=True,
    low_pass=None,
    high_pass=0.01,
    t_r=2,
    standardize=True,
    memory="nilearn_cache",
    memory_level=1,
    verbose=2,
)
masker.fit()

subject_time_series = []
func_filenames = rest_dataset.func
confound_filenames = rest_dataset.confounds
for func_filename, confound_filename in zip(
    func_filenames, confound_filenames
):
    print(f"Processing file {func_filename}")

    region_ts = masker.transform(func_filename, confounds=confound_filename)
    subject_time_series.append(region_ts)


##############################################################################
# Computing group-sparse precision matrices
# -----------------------------------------
from nilearn.connectome import GroupSparseCovarianceCV

gsc = GroupSparseCovarianceCV(verbose=2)
gsc.fit(subject_time_series)

try:
    from sklearn.covariance import GraphicalLassoCV
except ImportError:
    # for Scitkit-Learn < v0.20.0
    from sklearn.covariance import GraphLassoCV as GraphicalLassoCV

gl = GraphicalLassoCV(verbose=2)
gl.fit(np.concatenate(subject_time_series))


##############################################################################
# Displaying results
# ------------------
atlas_img = msdl_atlas_dataset.maps
atlas_region_coords = plotting.find_probabilistic_atlas_cut_coords(atlas_img)
labels = msdl_atlas_dataset.labels

plotting.plot_connectome(
    gl.covariance_,
    atlas_region_coords,
    edge_threshold="90%",
    title="Covariance",
    display_mode="lzr",
)
plotting.plot_connectome(
    -gl.precision_,
    atlas_region_coords,
    edge_threshold="90%",
    title="Sparse inverse covariance (GraphicalLasso)",
    display_mode="lzr",
    edge_vmax=0.5,
    edge_vmin=-0.5,
)
plot_matrices(gl.covariance_, gl.precision_, "GraphicalLasso", labels)

title = "GroupSparseCovariance"
plotting.plot_connectome(
    -gsc.precisions_[..., 0],
    atlas_region_coords,
    edge_threshold="90%",
    title=title,
    display_mode="lzr",
    edge_vmax=0.5,
    edge_vmin=-0.5,
)
plot_matrices(gsc.covariances_[..., 0], gsc.precisions_[..., 0], title, labels)

plotting.show()
