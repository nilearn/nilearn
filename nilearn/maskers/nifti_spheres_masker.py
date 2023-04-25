"""Transformer for computing seeds signals.

Mask nifti images by spherical volumes for seed-region analyses
"""
import warnings

import numpy as np
from joblib import Memory
from nilearn import image, masking
from nilearn._utils import CacheMixin, fill_doc, logger
from nilearn._utils.class_inspect import get_params
from nilearn._utils.niimg import img_data_dtype
from nilearn._utils.niimg_conversions import (
    _safe_get_data,
    check_niimg_3d,
    check_niimg_4d,
)
from nilearn.maskers.base_masker import BaseMasker, _filter_and_extract
from scipy import sparse
from sklearn import neighbors


def _apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap,
                                 mask_img=None):
    """Get only the rows which are occupied by sphere \
    at given seed locations and the provided radius.

    Rows are in target_affine and target_shape space.

    Parameters
    ----------
    seeds : List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as target_affine.

    niimg : 3D/4D Niimg-like object
        See :ref:`extracting_data`.
        Images to process.
        If a 3D niimg is provided, a singleton dimension will be added to
        the output to represent the single scan in the niimg.

    radius : float
        Indicates, in millimeters, the radius for the sphere around the seed.

    allow_overlap : boolean
        If False, a ValueError is raised if VOIs overlap

    mask_img : Niimg-like object, optional
        Mask to apply to regions before extracting signals. If niimg is None,
        mask_img is used as a reference space in which the spheres 'indices are
        placed.

    Returns
    -------
    X : 2D numpy.ndarray
        Signal for each brain voxel in the (masked) niimgs.
        shape: (number of scans, number of voxels)

    A : scipy.sparse.lil_matrix
        Contains the boolean indices for each sphere.
        shape: (number of seeds, number of voxels)

    """
    seeds = list(seeds)

    # Compute world coordinates of all in-mask voxels.
    if niimg is None:
        mask, affine = masking._load_mask_img(mask_img)
        # Get coordinate for all voxels inside of mask
        mask_coords = np.asarray(np.nonzero(mask)).T.tolist()
        X = None

    elif mask_img is not None:
        affine = niimg.affine
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(
            mask_img,
            target_affine=affine,
            target_shape=niimg.shape[:3],
            interpolation='nearest',
        )
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))

        X = masking._apply_mask_fmri(niimg, mask_img)

    elif niimg is not None:
        affine = niimg.affine
        if np.isnan(np.sum(_safe_get_data(niimg))):
            warnings.warn(
                'The imgs you have fed into fit_transform() contains NaN '
                'values which will be converted to zeroes.'
            )
            X = _safe_get_data(niimg, True).reshape([-1, niimg.shape[3]]).T
        else:
            X = _safe_get_data(niimg).reshape([-1, niimg.shape[3]]).T

        mask_coords = list(np.ndindex(niimg.shape[:3]))

    else:
        raise ValueError("Either a niimg or a mask_img must be provided.")

    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(image.resampling.coord_transform(
            sx, sy, sz, np.linalg.inv(affine)
        ))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = image.resampling.coord_transform(
        mask_coords[0], mask_coords[1], mask_coords[2], affine
    )
    mask_coords = np.asarray(mask_coords).T

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    for i, nearest in enumerate(nearests):
        if nearest is None:
            continue

        A[i, nearest] = True

    # Include the voxel containing the seed itself if not masked
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(list(map(int, seed)))] = True
        except ValueError:
            # seed is not in the mask
            pass

    sphere_sizes = np.asarray(A.tocsr().sum(axis=1)).ravel()
    empty_spheres = np.nonzero(sphere_sizes == 0)[0]
    if len(empty_spheres) != 0:
        raise ValueError(f'These spheres are empty: {empty_spheres}')

    if (not allow_overlap) and np.any(A.sum(axis=0) >= 2):
        raise ValueError('Overlap detected between spheres')

    return X, A


def _iter_signals_from_spheres(seeds, niimg, radius, allow_overlap,
                               mask_img=None):
    """Iterate over spheres.

    Parameters
    ----------
    seeds : :obj:`list` of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).

    niimg : 3D/4D Niimg-like object
        See :ref:`extracting_data`.
        Images to process.
        If a 3D niimg is provided, a singleton dimension will be added to
        the output to represent the single scan in the niimg.

    radius: float
        Indicates, in millimeters, the radius for the sphere around the seed.

    allow_overlap: boolean
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel).

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    """
    X, A = _apply_mask_and_get_affinity(seeds, niimg, radius,
                                        allow_overlap,
                                        mask_img=mask_img)
    for i, row in enumerate(A.rows):
        yield X[:, row]


class _ExtractionFunctor:

    func_name = 'nifti_spheres_masker_extractor'

    def __init__(self, seeds_, radius, mask_img, allow_overlap, dtype):
        self.seeds_ = seeds_
        self.radius = radius
        self.mask_img = mask_img
        self.allow_overlap = allow_overlap
        self.dtype = dtype

    def __call__(self, imgs):
        n_seeds = len(self.seeds_)
        imgs = check_niimg_4d(imgs, dtype=self.dtype)

        signals = np.empty(
            (imgs.shape[3], n_seeds),
            dtype=img_data_dtype(imgs)
        )
        for i, sphere in enumerate(_iter_signals_from_spheres(
                self.seeds_, imgs, self.radius, self.allow_overlap,
                mask_img=self.mask_img)):
            signals[:, i] = np.mean(sphere, axis=1)
        return signals, None


@fill_doc
class NiftiSpheresMasker(BaseMasker, CacheMixin):
    """Class for masking of Niimg-like objects using seeds.

    NiftiSpheresMasker is useful when data from given seeds should be
    extracted. Use case: Summarize brain signals from seeds that were
    obtained from prior knowledge.

    Parameters
    ----------
    seeds : :obj:`list` of triplet of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).

    radius : :obj:`float`, optional
        Indicates, in millimeters, the radius for the sphere around the seed.
        Default is None (signal is extracted on a single voxel).

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    allow_overlap : :obj:`bool`, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default=False.
    %(smoothing_fwhm)s
    %(standardize_maskers)s
    %(standardize_confounds)s
    high_variance_confounds : :obj:`bool`, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.
    %(detrend)s
    %(low_pass)s
    %(high_pass)s
    %(t_r)s
    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.
    %(memory)s
    %(memory_level1)s
    %(verbose0)s
    %(masker_kwargs)s

    Attributes
    ----------
    n_elements_ : :obj:`int`
        The number of seeds in the masker.

        .. versionadded:: 0.9.2

    seeds_ : :obj:`list` of :obj:`list`
        The coordinates of the seeds in the masker.

    See Also
    --------
    nilearn.maskers.NiftiMasker

    """

    # memory and memory_level are used by CacheMixin.

    def __init__(
        self,
        seeds,
        radius=None,
        mask_img=None,
        allow_overlap=False,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        high_variance_confounds=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        dtype=None,
        memory=Memory(location=None, verbose=0),
        memory_level=1,
        verbose=0,
        **kwargs,
    ):
        self.seeds = seeds
        self.mask_img = mask_img
        self.radius = radius
        self.allow_overlap = allow_overlap

        # Parameters for _smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.standardize_confounds = standardize_confounds
        self.high_variance_confounds = high_variance_confounds
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.dtype = dtype
        self.clean_kwargs = {
            k[7:]: v for k, v in kwargs.items() if k.startswith("clean__")
        }

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused; they are for scikit-learn compatibility.

        """
        if hasattr(self, 'seeds_'):
            return self

        error = (
            'Seeds must be a list of triplets of coordinates in '
            'native space.\n'
        )

        if not hasattr(self.seeds, '__iter__'):
            raise ValueError(
                error + 'Given seed list is of type: ' + type(self.seeds)
            )

        self.seeds_ = []
        # Check seeds and convert them to lists if needed
        for i, seed in enumerate(self.seeds):
            # Check the type first
            if not hasattr(seed, '__len__'):
                raise ValueError(
                    error + f'Seed #{i} is not a valid triplet '
                    f'of coordinates. It is of type {type(seed)}.'
                )
            # Convert to list because it is easier to process
            if isinstance(seed, np.ndarray):
                seed = seed.tolist()
            else:
                # in case of tuple
                seed = list(seed)

            # Check the length
            if len(seed) != 3:
                raise ValueError(
                    error + f'Seed #{i} is of length %{len(seed)} '
                    'instead of 3.'
                )

            self.seeds_.append(seed)

        self.n_elements_ = len(self.seeds_)

        return self

    def fit_transform(self, imgs, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.
            shape: (number of scans - number of volumes removed, )

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each sphere.
            shape: (number of scans, number of spheres)

        """
        return self.fit().transform(imgs, confounds=confounds,
                                    sample_mask=sample_mask)

    def _check_fitted(self):
        if not hasattr(self, "seeds_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform_single_imgs(self, imgs, confounds=None, sample_mask=None):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs : 3D/4D Niimg-like object
            See :ref:`extracting_data`.
            Images to process.
            If a 3D niimg is provided, a singleton dimension will be added to
            the output to represent the single scan in the niimg.

        confounds : CSV file or array-like or :obj:`pandas.DataFrame`, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        sample_mask : Any type compatible with numpy-array indexing, optional
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.
            shape: (number of scans - number of volumes removed, )

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each sphere.
            shape: (number of scans, number of spheres)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        self._check_fitted()

        params = get_params(NiftiSpheresMasker, self)
        params['clean_kwargs'] = self.clean_kwargs

        signals, _ = self._cache(
            _filter_and_extract,
            ignore=['verbose', 'memory', 'memory_level'])(
                imgs,
                _ExtractionFunctor(
                    self.seeds_, self.radius, self.mask_img,
                    self.allow_overlap, self.dtype
                ),
                # Pre-processing
                params,
                confounds=confounds,
                sample_mask=sample_mask,
                dtype=self.dtype,
                # Caching
                memory=self.memory,
                memory_level=self.memory_level,
                # kwargs
                verbose=self.verbose
        )
        return signals

    def inverse_transform(self, region_signals):
        """Compute voxel signals from spheres signals.

        Any mask given at initialization is taken into account. Throws an error
        if mask_img==None

        Parameters
        ----------
        region_signals : 1D/2D :obj:`numpy.ndarray`
            Signal for each region.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.

        Returns
        -------
        voxel_signals : :obj:`nibabel.nifti1.Nifti1Image`
            Signal for each sphere.
            shape: (mask_img, number of scans).

        """
        self._check_fitted()

        logger.log("computing image from signals", verbose=self.verbose)

        if self.mask_img is not None:
            mask = check_niimg_3d(self.mask_img)
        else:
            raise ValueError(
                'Please provide mask_img at initialization to '
                'provide a reference for the inverse_transform.'
            )

        _, adjacency = _apply_mask_and_get_affinity(
            self.seeds_, None, self.radius, self.allow_overlap, mask_img=mask
        )
        adjacency = adjacency.tocsr()
        # Compute overlap scaling for mean signal:
        if self.allow_overlap:
            n_adjacent_spheres = np.asarray(adjacency.sum(axis=0)).ravel()
            scale = 1 / np.maximum(1, n_adjacent_spheres)
            adjacency = adjacency.dot(sparse.diags(scale))

        img = adjacency.T.dot(region_signals.T).T
        return masking.unmask(img, self.mask_img)
