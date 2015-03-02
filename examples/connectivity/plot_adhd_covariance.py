"""
Sparse inverse covariance for a functional connectome
======================================================

This example shows how to extract signals from regions defined by an atlas,
and to estimate a functional connectome using a sparse inverse covariance
estimator.
"""
import matplotlib.pyplot as plt

from nilearn import plotting, image

def plot_matrices(cov, prec, title):
    """Plot covariance and precision matrices, for a given processing. """

    prec = prec.copy()  # avoid side effects

    # Put zeros on the diagonal, for graph clarity.
    size = prec.shape[0]
    prec[range(size), range(size)] = 0
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
adhd_dataset = datasets.fetch_adhd(n_subjects=1)


# Extracting region signals ###################################################
import nilearn.image
import nilearn.input_data

from sklearn.externals.joblib import Memory
mem = Memory('nilearn_cache')

masker = nilearn.input_data.NiftiMapsMasker(
    msdl_atlas_dataset.maps, resampling_target="maps", detrend=True,
    low_pass=None, high_pass=0.01, t_r=2.5, standardize=True,
    memory=mem, memory_level=1, verbose=2)
masker.fit()

fmri_filename = adhd_dataset.func[0]
confound_filename = adhd_dataset.confounds[0]

# Computing some confounds
hv_confounds = mem.cache(nilearn.image.high_variance_confounds)(
    fmri_filename)

time_series = masker.transform(fmri_filename,
                                confounds=[hv_confounds, confound_filename])


print("-- Computing graph-lasso inverse matrix ...")
from sklearn import covariance
gl = covariance.GraphLassoCV(verbose=2)
gl.fit(time_series)

# Displaying results ##########################################################
atlas_imgs = image.iter_img(msdl_atlas_dataset.maps)
atlas_region_coords = [plotting.find_xyz_cut_coords(img) for img in atlas_imgs]

title = "GraphLasso"
plotting.plot_connectome(-gl.precision_, atlas_region_coords,
                         edge_threshold='90%',
                         title="Sparse inverse covariance")
plotting.plot_connectome(gl.covariance_,
                         atlas_region_coords, edge_threshold='90%',
                         title="Covariance")
plot_matrices(gl.covariance_, gl.precision_, title)


plt.show()
