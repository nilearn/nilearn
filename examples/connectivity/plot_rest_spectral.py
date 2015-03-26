"""
Spectral clustering to segment brain from resting state fMRI
============================================================

We build an affinity matrix between voxels using Pearson's correlation
and then use it to segment the brain into functional regions.

Refernce :
    Craddock, R. Cameron, G.Andrew James, Paul E. Holtzheimer, Xiaoping P. Hu,
    and Helen S. Mayberg. "A Whole Brain fMRI Atlas Generated via Spatially
    Constrained Spectral Clustering". Human Brain Mapping 33, no 8 (2012):
    1914-1928. doi:10.1002/hbm.21333.
"""
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph

import nibabel as nib

### Fetch and mask data #######################################################
print("Loading resting-state data and masking subject data...")

from nilearn import datasets, input_data

dataset = datasets.fetch_adhd(n_subjects=1)  # only data from first subject

# We restrict ourselves to one hemisphere to speed up computation
func_nii = nib.load(dataset.func[0])
data_left = func_nii.get_data()
data_left[:(data_left.shape[0] / 2), :, :, :] = 0
lh_img = nib.Nifti1Image(data_left, func_nii.get_affine())

nifti_masker = input_data.NiftiMasker(
    smoothing_fwhm=0., standardize=False,
    detrend=True,  # remove linear trends in the time/4th dimension
    memory='nilearn_cache', memory_level=100, verbose=10)
X = nifti_masker.fit_transform(lh_img)  # get only fMRI signals from left side
mask = nifti_masker.mask_img_.get_data().astype(np.bool)  # convert to 0/1

from nilearn.image import index_img
from nilearn.plotting import plot_roi, plot_stat_map
plot_stat_map(index_img(nifti_masker.inverse_transform(X), -1),
              title='Examplary volume (restricted to left hemipshere)')

### Affinity Matrix ##########################################################
print("Computing affinity matrix...")

# Set up persistence framework.  Using 'partial' allows us to make the
# cache function be called with the same memory and verbose parameters
# every time, making for more consistent, and shorter, calls.
import functools
from nilearn._utils.cache_mixin import cache
my_cache_fn = functools.partial(cache, memory='nilearn_cache', verbose=10)

# Compute the connectivity graph (it is sparse)
def compute_affinity(X, mask):
    connectivity = grid_to_graph(*mask.shape, mask=mask)

    # Compute the Pearson correlation matrix from data and connectivity
    rows, cols = connectivity.nonzero()
    values = np.zeros(rows.shape)
    for xi, (r, c) in enumerate(zip(rows, cols)):
        values[xi] = sp.stats.pearsonr(X[:, r], X[:, c])[0]

    # Keep a number of correlation equal to XX% of the number of voxels
    n_voxels = connectivity.shape[0]
    thr = np.sort(values)[-int(n_voxels * 1.8)]
    rows = rows[np.where(values >= thr)]
    cols = cols[np.where(values >= thr)]
    values = values[np.where(values >= thr)]

    pct_kept = len(values) * 100. / connectivity.nnz
    pct_possible = len(values) * 100. / n_voxels**2
    print("Voxels: %d; cutoff: %.4f; %% kept %.2f, %% possible: %.4f"
          % (n_voxels, thr, pct_kept, pct_possible))

    affinity = sp.sparse.coo_matrix((values, (rows, cols)))
    return affinity, connectivity
affinity, connectivity = my_cache_fn(compute_affinity)(X, mask)

plot_stat_map(nifti_masker.inverse_transform(np.array(connectivity.sum(0))))
plt.show()

### Spectral clustering #######################################################
print("Running spectral clustering...")

from sklearn.cluster import spectral_clustering
clustering = my_cache_fn(spectral_clustering)(affinity, n_clusters=50)

### Plot results ##############################################################
print("Plotting the results...")
cluster_img = nifti_masker.inverse_transform(clustering)
plot_roi(cluster_img)
plt.show()
