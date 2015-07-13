"""
Transformer for computing seeds signals.
=======
Mask nifti images by spherical volumes for seed-region analyses
"""
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import neighbors
from sklearn.externals.joblib import Memory

from .. import _utils
from .._utils import logger, CacheMixin
from .._utils.niimg_conversions import check_niimg, check_niimg_3d
from .. import signal
from .. import image
from .. import masking
from distutils.version import LooseVersion


def _iter_signals_from_spheres(seeds, niimg, radius, mask_img=None):
    seeds = list(seeds)
    niimg = check_niimg(niimg)
    affine = niimg.get_affine()

    # Compute world coordinates of all in-mask voxels.

    if mask_img is not None:
        mask_img = check_niimg_3d(mask_img)
        mask_img = image.resample_img(mask_img, target_affine=affine,
                                      target_shape=niimg.shape[:3],
                                      interpolation='nearest')
        mask, _ = masking._load_mask_img(mask_img)
        mask_coords = list(np.where(mask != 0))

        X = masking._apply_mask_fmri(niimg, mask_img)
    else:
        mask_coords = list(zip(*np.ndindex(niimg.shape[:3])))
        X = niimg.get_data().reshape([-1, niimg.shape[3]]).T
    mask_coords.append(np.ones(len(mask_coords[0]), dtype=np.int))
    mask_coords = np.asarray(mask_coords)
    mask_coords = np.dot(affine, mask_coords)[:3].T

    if (radius is not None and
            LooseVersion(sklearn.__version__) < LooseVersion('0.16')):
        # Fix for scikit learn versions below 0.16. See
        # https://github.com/scikit-learn/scikit-learn/issues/4072
        radius += 1e-6

    clf = neighbors.NearestNeighbors(radius=radius)
    A = clf.fit(mask_coords).radius_neighbors_graph(seeds)
    A = A.tolil()
    # Include selfs
    mask_coords = mask_coords.astype(int).tolist()
    for i, seed in enumerate(seeds):
        try:
            A[i, mask_coords.index(seed)] = True
        except ValueError:
            # seed is not in the mask
            pass
    del mask_coords

    for i, row in enumerate(A.rows):
        if len(row) == 0:
            raise ValueError('Sphere around seed #%i is empty' % i)
        yield X[:, row]


def _signals_from_spheres(seeds, niimg, radius, mask_img=None):
    seeds = list(seeds)
    n_seeds = len(seeds)

    signals = np.empty((niimg.shape[3], n_seeds))
    for i, sphere in enumerate(_iter_signals_from_spheres(
            seeds, niimg, radius, mask_img=mask_img)):
        signals[:, i] = np.mean(sphere, axis=1)

    return signals


class NiftiSpheresMasker(BaseEstimator, TransformerMixin, CacheMixin):
    """Class for masking of Niimg-like objects using seeds.

    NiftiSpheresMasker is useful when data from given seeds should be
    extracted. Use case: Summarize brain signals from seeds that were
    obtained from prior knowledge.

    Parameters
    ==========
    seeds: List of triplet of coordinates in native space
        Seed definitions. List of coordinates of the seeds in the same space
        as the images (typically MNI or TAL).

    radius: float, optional
        Indicates, in millimeters, the radius for the sphere around the seed.
        Default is None (signal is extracted on a single voxel).

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Mask to apply to regions before extracting signals.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the full-width half maximum in
        millimeters of the spatial smoothing to apply to the signal.

    standardize: boolean, optional
        If standardize is True, the time-series are centered and normed:
        their mean is set to 0 and their variance to 1 in the time dimension.

    detrend: boolean, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    low_pass: False or float, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details.

    high_pass: False or float, optional
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
    ========
    nilearn.input_data.NiftiMasker
    """
    # memory and memory_level are used by CacheMixin.

    def __init__(self, seeds, radius=None, mask_img=None,
                 smoothing_fwhm=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=1,
                 verbose=0):
        self.seeds = seeds
        self.mask_img = mask_img
        self.radius = radius

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
                seed = seed.to_list()
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
        return self.fit().transform(imgs, confounds=confounds)

    def _check_fitted(self):
        if not hasattr(self, "seeds_"):
            raise ValueError('It seems that %s has not been fitted. '
                             'You must call fit() before calling transform().'
                             % self.__class__.__name__)

    def transform(self, imgs, confounds=None):
        """Extract signals from Nifti-like objects.

        Parameters
        ==========
        imgs: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Images to process. It must boil down to a 4D image with scans
            number as last dimension.

        confounds: array-like, optional
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        =======
        signals: 2D numpy.ndarray
            Signal for each region.
            shape: (number of scans, number of regions)

        """
        self._check_fitted()

        logger.log("loading images: %s" %
                   _utils._repr_niimgs(imgs)[:200], verbose=self.verbose)
        imgs = _utils.check_niimg(imgs)

        if self.smoothing_fwhm is not None:
            logger.log("smoothing images", verbose=self.verbose)
            imgs = self._cache(image.smooth_img)(
                imgs, fwhm=self.smoothing_fwhm)

        logger.log("extracting region signals", verbose=self.verbose)
        signals = self._cache(_signals_from_spheres)(
                self.seeds_, imgs, radius=self.radius, mask_img=self.mask_img)

        logger.log("cleaning extracted signals", verbose=self.verbose)
        signals = self._cache(signal.clean
                                     )(signals,
                                       detrend=self.detrend,
                                       standardize=self.standardize,
                                       t_r=self.t_r,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       confounds=confounds)
        return signals
