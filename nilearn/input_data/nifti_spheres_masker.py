"""
Transformer for computing seeds signals
----------------------------------------

Mask nifti images by spherical volumes for seed-region analyses
"""
import numpy as np
import sklearn
from sklearn import neighbors
from sklearn.externals.joblib import Memory
from distutils.version import LooseVersion

from ..image.resampling import coord_transform
from .._utils import CacheMixin
from .._utils.niimg_conversions import check_niimg_4d, check_niimg_3d
from .._utils.class_inspect import get_params
from .._utils.compat import get_affine
from .. import image
from .. import masking
from .base_masker import filter_and_extract, BaseMasker


def _apply_mask_and_get_affinity(seeds, niimg, radius, allow_overlap,
                                 mask_img=None):
    seeds = list(seeds)
    affine = get_affine(niimg)

    # Compute world coordinates of all in-mask voxels.

    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(mask_img, target_affine=affine,
                                      target_shape=niimg.shape[:3],
                                      interpolation='nearest')
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(zip(*np.where(mask != 0)))

        X = masking._apply_mask_fmri(niimg, mask_img)
    else:
        mask_coords = list(np.ndindex(niimg.shape[:3]))
        X = niimg.get_data().reshape([-1, niimg.shape[3]]).T

    # For each seed, get coordinates of nearest voxel
    nearests = []
    for sx, sy, sz in seeds:
        nearest = np.round(coord_transform(sx, sy, sz, np.linalg.inv(affine)))
        nearest = nearest.astype(int)
        nearest = (nearest[0], nearest[1], nearest[2])
        try:
            nearests.append(mask_coords.index(nearest))
        except ValueError:
            nearests.append(None)

    mask_coords = np.asarray(list(zip(*mask_coords)))
    mask_coords = coord_transform(mask_coords[0], mask_coords[1],
                                  mask_coords[2], affine)
    mask_coords = np.asarray(mask_coords).T

    if (radius is not None and
            LooseVersion(sklearn.__version__) < LooseVersion('0.16')):
        # Fix for scikit learn versions below 0.16. See
        # https://github.com/scikit-learn/scikit-learn/issues/4072
        radius += 1e-6

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
            A[i, mask_coords.index(seed)] = True
        except ValueError:
            # seed is not in the mask
            pass

    if not allow_overlap:
        if np.any(A.sum(axis=0) >= 2):
            raise ValueError('Overlap detected between spheres')

    return X, A


def _iter_signals_from_spheres(seeds, niimg, radius, allow_overlap,
                               mask_img=None):
    """Utility function to iterate over spheres.
    Parameters
    ----------
    seeds: List of triplets of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).
    imgs: 3D/4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Images to process. It must boil down to a 4D image with scans
        number as last dimension.
    radius: float
        Indicates, in millimeters, the radius for the sphere around the seed.
    allow_overlap: boolean
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel).
    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to regions before extracting signals.
    """
    X, A = _apply_mask_and_get_affinity(seeds, niimg, radius,
                                        allow_overlap,
                                        mask_img=mask_img)
    for i, row in enumerate(A.rows):
        if len(row) == 0:
            raise ValueError('Sphere around seed #%i is empty' % i)
        yield X[:, row]


class _ExtractionFunctor(object):

    func_name = 'nifti_spheres_masker_extractor'

    def __init__(self, seeds_, radius, mask_img, allow_overlap):
        self.seeds_ = seeds_
        self.radius = radius
        self.mask_img = mask_img
        self.allow_overlap = allow_overlap

    def __call__(self, imgs):
        n_seeds = len(self.seeds_)
        imgs = check_niimg_4d(imgs)

        signals = np.empty((imgs.shape[3], n_seeds))
        for i, sphere in enumerate(_iter_signals_from_spheres(
                self.seeds_, imgs, self.radius, self.allow_overlap,
                mask_img=self.mask_img)):
            signals[:, i] = np.mean(sphere, axis=1)

        return signals, None


class NiftiSpheresMasker(BaseMasker, CacheMixin):
    """Class for masking of Niimg-like objects using seeds.

    NiftiSpheresMasker is useful when data from given seeds should be
    extracted. Use case: Summarize brain signals from seeds that were
    obtained from prior knowledge.

    Parameters
    ----------
    seeds: List of triplet of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).

    radius: float, optional
        Indicates, in millimeters, the radius for the sphere around the seed.
        Default is None (signal is extracted on a single voxel).

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to regions before extracting signals.

    allow_overlap: boolean, optional
        If False, an error is raised if the maps overlaps (ie at least two
        maps have a non-zero value for the same voxel). Default is False.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is set to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    low_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    high_pass: None or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    t_r: float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    memory: joblib.Memory or str, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level: int, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    verbose: integer, optional
        Indicate the level of verbosity. By default, nothing is printed.

    See also
    --------
    nilearn.input_data.NiftiMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, seeds, radius=None, mask_img=None, allow_overlap=False,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 verbose=0):
        self.seeds = seeds
        self.mask_img = mask_img
        self.radius = radius
        self.allow_overlap = allow_overlap

        # Parameters for _smooth_array
        self.smoothing_fwhm = smoothing_fwhm

        # Parameters for clean()
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level

        self.verbose = verbose

    def fit(self, X=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.
        """
        if hasattr(self, 'seeds_'):
            return self

        error = ("Seeds must be a list of triplets of coordinates in "
                 "native space.\n")

        if not hasattr(self.seeds, '__iter__'):
            raise ValueError(error + "Given seed list is of type: " +
                             type(self.seeds))

        self.seeds_ = []
        # Check seeds and convert them to lists if needed
        for i, seed in enumerate(self.seeds):
            # Check the type first
            if not hasattr(seed, '__len__'):
                raise ValueError(error + "Seed #%i is not a valid triplet "
                                 "of coordinates. It is of type %s."
                                 % (i, type(seed)))
            # Convert to list because it is easier to process
            if isinstance(seed, np.ndarray):
                seed = seed.tolist()
            else:
                # in case of tuple
                seed = list(seed)

            # Check the length
            if len(seed) != 3:
                raise ValueError(error + "Seed #%i is of length %i "
                                 "instead of 3." % (i, len(seed)))

            self.seeds_.append(seed)

        return self

    def fit_transform(self, imgs, confounds=None):
        """Prepare and perform signal extraction"""
        return self.fit().transform(imgs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "seeds_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform_single_imgs(self, imgs, confounds=None):
        """Extract signals from a single 4D niimg.

        Parameters
        ----------
        imgs: 3D/4D Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds: CSV file or array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        -------
        region_signals: 2D numpy.ndarray
            Signal for each sphere.
            shape: (number of scans, number of spheres)
        """
        self._check_fitted()

        params = get_params(NiftiSpheresMasker, self)

        signals, _ = self._cache(
                filter_and_extract,
                ignore=['verbose', 'memory', 'memory_level'])(
            # Images
            imgs, _ExtractionFunctor(self.seeds_, self.radius, self.mask_img,
                                     self.allow_overlap),
            # Pre-processing
            params,
            confounds=confounds,
            # Caching
            memory=self.memory,
            memory_level=self.memory_level,
            # kwargs
            verbose=self.verbose)

        return signals
