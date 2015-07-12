"""
Group Sparse inverse covariance for multi-subject connectome
=============================================================

This example shows how to estimate a connectome on a groupe of subjects
using the group sparse inverse covariance estimate.

"""
import matplotlib.pyplot as plt
import numpy as np

from nilearn import plotting


n_subjects = 4  # subjects to consider for group-sparse covariance (max: 40)


def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[list(range(size)), list(range(size))] = 0
    span = max(abs(prec.min()), abs(prec.max()))

    # Display covariance matrix
    plt.figure()
    plt.imshow(cov, interpolation="nearest",
               vmin=-1, vmax=1, cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title("%s / covariance" % title)

    # Display precision matrix
    plt.figure()
    plt.imshow(prec, interpolation="nearest",
               vmin=-span, vmax=span,
               cmap=plotting.cm.bwr)
    plt.colorbar()
    plt.title("%s / precision" % title)


# Fetching datasets ###########################################################
print("-- Fetching datasets ...")
from nilearn import datasets
msdl_atlas_dataset = datasets.fetch_msdl_atlas()
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data


# Extracting region signals ###################################################
from nilearn import image
from nilearn import input_data

from sklearn.externals.joblib import Memory
mem = Memory('nilearn_cache')

masker = input_data.NiftiMapsMasker(
    msdl_atlas_dataset.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

subject_time_series = []
func_filenames = adhd_dataset.func
confound_filenames = adhd_dataset.confounds
for func_filename, confound_filename in zip(func_filenames,
                                            confound_filenames):
    print("Processing file %s" % func_filename)

    # Computing some confounds
    hv_confounds = mem.cache(image.high_variance_confounds)(
        func_filename)

    region_ts = masker.transform(func_filename,
                                 confounds=[hv_confounds, confound_filename])
    subject_time_series.append(region_ts)

# Computing group-sparse precision matrices ###################################
print("-- Computing group-sparse precision matrices ...")
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
gsc = GroupSparseCovarianceCV(verbose=2)
gsc.fit(subject_time_series)

print("-- Computing graph-lasso precision matrices ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(verbose=2)
gl.fit(np.concatenate(subject_time_series))

# Displaying results ##########################################################
atlas_imgs = image.iter_img(msdl_atlas_dataset.maps)
atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in atlas_imgs]

title = "GraphLasso"
plotting.plot_connectome(-gl.precision_, atlas_region_coords,
                         edge_threshold='90%',
                         title="Sparse inverse covariance (GraphLasso)")
plotting.plot_connectome(gl.covariance_,
                         atlas_region_coords, edge_threshold='90%',
                         title="Covariance")
plot_matrices(gl.covariance_, gl.precision_, title)

title = "GroupSparseCovariance"
plotting.plot_connectome(-gsc.precisions_[..., 0],
                         atlas_region_coords, edge_threshold='90%',
                         title=title)
plot_matrices(gsc.covariances_[..., 0],
              gsc.precisions_[..., 0], title)

plt.show()
