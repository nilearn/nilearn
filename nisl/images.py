"""
Some tools for processing images and masks.

Authors : Vincent Michel (vm.michel@gmail.com)
          Alexandre Gramfort (alexandre.gramfort@inria.fr)
License: BSD 3 clause
"""
import numpy as np
from scipy import sparse

from sklearn import neighbors


def adjacency_from_mask(mask, radius=5):
    """
    Construct an adjacency image from a mask.

    Parameters
    ----------
    mask: a 3d boolean mask
    radius: radius of the KNN search to be used
    (depends to the resolution of the mask. Default is 5)
    Return
    ------
    A : sparse matrix.
    adjacency matrix. Defines for each sample the neigbhoring samples
    following a given structure of the data.
    """
    flat_mask = np.array(np.where(mask))
    A = sparse.lil_matrix((flat_mask.shape[1], flat_mask.shape[1]))
    clf = neighbors.NearestNeighbors(radius=radius)
    dist, ind = clf.fit(flat_mask.T).kneighbors(flat_mask.T)
    for i, li in enumerate(ind):
        A[i, list(li[1:])] = np.ones(len(li[1:]))
    return A
