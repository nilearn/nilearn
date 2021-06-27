"""Methods For defining voxelwise connectivity"""
# Author: Derek Pisner

import numpy as np
from scipy import divide, prod
from .._utils.niimg import _safe_get_data

def indx_1dto3d(idx, sz):
    """
    Translate 1D vector coordinates to 3D matrix coordinates for a 3D matrix
    of size sz.

    Parameters
    ----------
    idx : array
        A 1D numpy coordinate vector.
    sz : array
        Shape of 3D matrix idx.

    Returns
    -------
    x : int
        x-coordinate of 3D matrix coordinates.
    y : int
        y-coordinate of 3D matrix coordinates.
    z : int
        z-coordinate of 3D matrix coordinates.
    """

    x = divide(idx, prod(sz[1:3]))
    y = divide(idx - x * prod(sz[1:3]), sz[2])
    z = idx - x * prod(sz[1:3]) - y * sz[2]
    return x, y, z


def indx_3dto1d(idx, sz):
    """
    Translate 3D matrix coordinates to 1D vector coordinates for a 3D matrix
    of size sz.

    Parameters
    ----------
    idx : array
        A 3D numpy array of matrix coordinates.
    sz : array
        Shape of 3D matrix idx.

    Returns
    -------
    idx1 : array
        A 1D numpy coordinate vector.
    """

    if np.linalg.matrix_rank(idx) == 1:
        idx1 = idx[0] * prod(sz[1:3]) + idx[1] * sz[2] + idx[2]
    else:
        idx1 = idx[:, 0] * prod(sz[1:3]) + idx[:, 1] * sz[2] + idx[:, 2]
    return idx1


def make_local_connectivity_scorr(func_img, clust_mask_img, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    spatial correlation between the whole brain FC maps generated from the
    time series from voxel i and voxel j. Connectivity is only calculated
    between a voxel and the 27 voxels in its 3D neighborhood
    (face touching and edge touching).

    Parameters
    ----------
    func_img : Nifti1Image
        4D Nifti1Image containing fMRI data.
    clust_mask_img : Nifti1Image
        3D NIFTI file containing a mask, which restricts the voxels used in
        the analysis.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    Returns
    -------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the spatial
        correlation between the time series from voxel i and voxel j

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333

    """
    from scipy.sparse import csc_matrix
    from itertools import product

    neighbors = np.array(
        sorted(
            sorted(
                sorted(
                    [list(x) for x in list(set(product({-1, 0, 1},
                                                       repeat=3)))],
                    key=lambda k: (k[0]),
                ),
                key=lambda k: (k[1]),
            ),
            key=lambda k: (k[2]),
        )
    )

    # Read in the mask
    msz = clust_mask_img.shape

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(
        np.asarray(
            clust_mask_img.dataobj).astype("bool"),
        prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = _safe_get_data(func_img).astype(np.float32)
    func_data = np.reshape(func_data, (prod(sz[:3]), sz[3]))

    # Mask the datset to only the in-mask voxels
    func_data = func_data[iv, :]
    func_data_sz = func_data.shape

    # Z-score fmri time courses, this makes calculation of the
    # correlation coefficient a simple matrix product
    func_data_s = np.tile(np.std(func_data, 1), (func_data_sz[1], 1)).T

    # Replace 0 with large number to avoid div by zero
    func_data_s[func_data_s == 0] = 1000000
    func_data_m = np.tile(np.mean(func_data, 1), (func_data_sz[1], 1)).T
    func_data = (func_data - func_data_m) / func_data_s

    # Set values with no variance to zero
    func_data[func_data_s == 0] = 0
    func_data[np.isnan(func_data)] = 0

    # Remove voxels with zero variance, do this here so that the mapping will
    # be consistent across subjects
    vndx = np.nonzero(np.var(func_data, 1) != 0)[0]
    iv = iv[vndx]
    m = len(iv)
    print(m, " # of non-zero valued or non-zero variance voxels in the mask")

    # Construct a sparse matrix from the mask
    msk = csc_matrix(
        (vndx + 1, (iv, np.zeros(m))), shape=(prod(msz), 1), dtype=np.float32
    )

    sparse_i = []
    sparse_j = []
    sparse_w = [[]]

    for i in range(0, m):
        if i % 1000 == 0:
            print("voxel #", i)

        # Convert index into 3D and calculate neighbors, then convert resulting
        # 3D indices into 1D
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])

        # Convert 1D indices into masked versions
        ondx1d = msk[ndx1d].todense()

        # Exclude indices not in the mask
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]].flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]])
        ondx1d = ondx1d.flatten() - 1

        # Keep track of the index corresponding to the "seed"
        nndx = np.nonzero(ndx1d == iv[i])[0]

        # Extract the time courses corresponding to the "seed" and 3D
        # neighborhood voxels
        tc = np.array(func_data[ondx1d.astype("int"), :])

        # Ensure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # Calculate functional connectivity maps for "seed" and 3D neighborhood
        # voxels
        R = np.corrcoef(np.dot(tc, func_data.T) / (sz[3] - 1))

        if np.linalg.matrix_rank(R) == 1:
            R = np.reshape(R, (1, 1))

        # Set nans to 0
        R[np.isnan(R)] = 0

        # Set values below thresh to 0
        R[R < thresh] = 0

        # Calculate the spatial correlation between FC maps
        if np.linalg.matrix_rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Keep track of the indices and the correlation weights to construct
        # sparse connectivity matrix
        sparse_i = np.append(sparse_i, ondx1d, 0)
        sparse_j = np.append(sparse_j, (ondx1d[nndx]) * np.ones(len(ondx1d)))
        sparse_w = np.append(sparse_w, R[nndx, :], 1)

    # Ensure that the weight vector is the correct shape
    sparse_w = np.reshape(sparse_w, prod(np.shape(sparse_w)))

    # Concatenate the i, j, and w_ij vectors
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(outlist) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    outlist = np.reshape(outlist, (3, int(n)))

    m = max(max(outlist[0, :]), max(outlist[1, :])) + 1

    W = csc_matrix(
        (outlist[2, :], (outlist[0, :], outlist[1, :])),
        shape=(int(m), int(m)),
        dtype=np.float32,
    )

    return W


def make_local_connectivity_tcorr(func_img, clust_mask_img, thresh):
    """
    Constructs a spatially constrained connectivity matrix from a fMRI dataset.
    The weights w_ij of the connectivity matrix W correspond to the
    temporal correlation between the time series from voxel i and voxel j.
    Connectivity is only calculated between a voxel and the 27 voxels in its 3D
    neighborhood (face touching and edge touching).

    Parameters
    ----------
    func_img : Nifti1Image
        4D Nifti1Image containing fMRI data.
    clust_mask_img : Nifti1Image
        3D NIFTI file containing a mask, which restricts the
        voxels used in the analysis.
    thresh : str
        Threshold value, correlation coefficients lower than this value
        will be removed from the matrix (set to zero).

    Returns
    -------
    W : Compressed Sparse Matrix
        A Scipy sparse matrix, with weights corresponding to the temporal
        correlation between the time series from voxel i and voxel j

    References
    ----------
    .. [1] Craddock, R. C., James, G. A., Holtzheimer, P. E., Hu, X. P., &
      Mayberg, H. S. (2012). A whole brain fMRI atlas generated via
      spatially constrained spectral clustering. Human Brain Mapping.
      https://doi.org/10.1002/hbm.21333

    """
    from scipy.sparse import csc_matrix
    from itertools import product

    # Index array used to calculate 3D neigbors
    neighbors = np.array(
        sorted(
            sorted(
                sorted(
                    [list(x) for x in list(set(product({-1, 0, 1},
                                                       repeat=3)))],
                    key=lambda k: (k[0]),
                ),
                key=lambda k: (k[1]),
            ),
            key=lambda k: (k[2]),
        )
    )

    # Read in the mask
    msz = np.shape(np.asarray(clust_mask_img.dataobj).astype("bool"))

    # Convert the 3D mask array into a 1D vector
    mskdat = np.reshape(
        np.asarray(
            clust_mask_img.dataobj).astype("bool"),
        prod(msz))

    # Determine the 1D coordinates of the non-zero elements of the mask
    iv = np.nonzero(mskdat)[0]
    m = len(iv)
    print(f"\nTotal non-zero voxels in the mask: {m}\n")
    sz = func_img.shape

    # Reshape fmri data to a num_voxels x num_timepoints array
    func_data = func_img.get_fdata(dtype=np.float32)
    func_data = np.reshape(func_data, (prod(sz[:3]), sz[3]))

    # Construct a sparse matrix from the mask
    msk = csc_matrix(
        (list(range(1, m + 1)), (iv, np.zeros(m))),
        shape=(prod(sz[:-1]), 1),
        dtype=np.float32,
    )
    sparse_i = []
    sparse_j = []
    sparse_w = []

    negcount = 0

    # Loop over all of the voxels in the mask
    print("Voxels:")
    for i in range(0, m):
        if i % 1000 == 0:
            print(str(i))
        # Calculate the voxels that are in the 3D neighborhood of the center
        # voxel
        ndx1d = indx_3dto1d(indx_1dto3d(iv[i], sz[:-1]) + neighbors, sz[:-1])

        # Restrict the neigborhood using the mask
        ondx1d = msk[ndx1d].todense()
        ndx1d = ndx1d[np.nonzero(ondx1d)[0]].flatten()
        ondx1d = np.array(ondx1d[np.nonzero(ondx1d)[0]]).flatten()

        # Determine the index of the seed voxel in the neighborhood
        nndx = np.nonzero(ndx1d == iv[i])[0]

        # Extract the timecourses for all of the voxels in the neighborhood
        tc = np.array(func_data[ndx1d.astype("int"), :])

        # Ensure that the "seed" has variance, if not just skip it
        if np.var(tc[nndx, :]) == 0:
            continue

        # Calculate the correlation between all of the voxel TCs
        R = np.corrcoef(tc)

        if np.linalg.matrix_rank(R) == 1:
            R = np.reshape(R, (1, 1))

        # Set nans to 0
        R[np.isnan(R)] = 0

        # Set values below thresh to 0
        R[R < thresh] = 0

        if np.linalg.matrix_rank(R) == 0:
            R = np.reshape(R, (1, 1))

        # Extract just the correlations with the seed TC
        R = R[nndx, :].flatten()

        # Set NaN values to 0
        negcount = negcount + sum(R < 0)

        # Determine the non-zero correlations (matrix weights) and add their
        # indices and values to the list
        nzndx = np.nonzero(R)[0]
        if len(nzndx) > 0:
            sparse_i = np.append(sparse_i, ondx1d[nzndx] - 1, 0)
            sparse_j = np.append(sparse_j,
                                 (ondx1d[nndx] - 1) * np.ones(len(nzndx)))
            sparse_w = np.append(sparse_w, R[nzndx], 0)

    # Concatenate the i, j and w_ij into a single vector
    outlist = sparse_i
    outlist = np.append(outlist, sparse_j)
    outlist = np.append(outlist, sparse_w)

    # Calculate the number of non-zero weights in the connectivity matrix
    n = len(outlist) / 3

    # Reshape the 1D vector read in from infile in to a 3xN array
    outlist = np.reshape(outlist, (3, int(n)))

    m = max(max(outlist[0, :]), max(outlist[1, :])) + 1

    W = csc_matrix(
        (outlist[2, :], (outlist[0, :], outlist[1, :])),
        shape=(int(m), int(m)),
        dtype=np.float32,
    )

    return W
