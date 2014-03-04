"""
Spectral clustering to segment brain from resting state fMRI
============================================================

We build an affinity matrix between voxels using Pearson correlation coefficient
and then use it to segment the brain into functional regions.
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
                              smoothing_fwhm=6., standardize=False)
X = nifti_masker.fit_transform(dataset.func[0])
mask = nifti_masker.mask_img_.get_data().astype(np.bool)

### Spectral clustering #######################################################

print 'Computing affinity matrix... '
t0 = time.time()
# Compute the connecitvity graph (it is sparse)
connectivity = grid_to_graph(*mask.shape, mask=mask)

# Compute the Pearson correlation matrix from data and connectivity
rows, cols = connectivity.nonzero()
values = np.zeros(rows.shape)
for i, (r, c) in enumerate(zip(rows, cols)):
    corr = sp.stats.pearsonr(X[:, r], X[:, c])[0]
    if np.isnan(corr):
        continue
    if corr <= 0.5:
        # Sparsify matrix
        corr = 0.
    values[i] = corr
correlation = sp.sparse.coo_matrix((values, (rows, cols)))
print '... done (%.2fs)' % (time.time() - t0)

# Apply spectral clustering on correlation matrix
from sklearn.cluster import spectral_clustering

print 'Running spectral clustering... '
t0 = time.time()
clustering = spectral_clustering(correlation,
        n_clusters=50, assign_labels='discretize')
print '... done (%.2fs)' % (time.time() - t0)

clustering = nifti_masker.inverse_transform(clustering)

plt.imshow(clustering.get_data()[..., 25] + 1, cmap='Set1',
          interpolation='nearest')
plt.show()
