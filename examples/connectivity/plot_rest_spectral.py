"""
Spectral clustering to segment brain from resting state fMRI
============================================================

We build an affinity matrix between voxels using Pearson correlation
coefficient and then use it to segment the brain into functional regions.

Refernce :
    Craddock, R. Cameron, G.Andrew James, Paul E. Holtzheimer, Xiaoping P. Hu,
    and Helen S. Mayberg. "A Whole Brain fMRI Atlas Generated via Spatially
    Constrained Spectral Clustering". Human Brain Mapping 33, no 8 (2012):
    1914-1928. doi:10.1002/hbm.21333.
"""

from sklearn.feature_extraction.image import grid_to_graph

### Load adhd dataset #########################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from nilearn import datasets, input_data
import time


### Fetch and mask data #######################################################

dataset = datasets.fetch_adhd(n_subjects=1)
nifti_masker = input_data.NiftiMasker(memory='nilearn_cache', memory_level=1,
                              smoothing_fwhm=12., standardize=False)
X = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)
print nifti_masker.mask_img_.get_affine()

### Spectral clustering #######################################################

print 'Computing affinity matrix... '
# Compute the connectivity graph (it is sparse)
connectivity = grid_to_graph(*mask.shape, mask=mask)

# Compute the Pearson correlation matrix from data and connectivity
rows, cols = connectivity.nonzero()
values = np.zeros(rows.shape)
for i, (r, c) in enumerate(zip(rows, cols)):
    corr = sp.stats.pearsonr(X[:, r], X[:, c])[0]
    if np.isnan(corr):
        values[i] = 0.
        continue
    #if corr <= 0.9:
        # Sparsify matrix
        #corr = 0.
    values[i] = corr
# Keep a number of correlation equal to XX% of the number of voxels
n_voxels = connectivity.shape[0]
thr = np.sort(values)[- int(n_voxels * 1.8)]
print n_voxels, thr
rows = rows[np.where(values >= thr)]
cols = cols[np.where(values >= thr)]
values = values[np.where(values >= thr)]

affinity = sp.sparse.coo_matrix((values, (rows, cols)))
# End computing affinity matrix

print 'Running spectral clustering... '
# Apply spectral clustering on affinity matrix
from sklearn.cluster import spectral_clustering
clustering = spectral_clustering(affinity,
        n_clusters=20, assign_labels='discretize')
# End spectral clustering

clustering = nifti_masker.inverse_transform(clustering)

plt.imshow(clustering.get_data()[..., 25] + 1, cmap='Set1',
          interpolation='nearest')
plt.show()
