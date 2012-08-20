"""
Transformer used to apply basic tranisformations on MRI data.
"""

import types
import collections
import itertools

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

import nibabel

from . import masking
from . import resampling
from . import preprocessing
from . import utils

def _check_nifti_methods(object):
    try:
        get_data = getattr(object, "get_data")
        get_affine = getattr(object, "get_affine")
        return callable(get_data) and callable(get_affine)
    except Exception:
        return False


def check_niimg(data):
    if isinstance(data, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(data)
    else:
        # it is an object, it should have get_data and get_affine methods
        if not _check_nifti_methods(data):
            raise AttributeError("Given data does not expose"
                " get_data or get_affine methods")
        result = data
    return result


def collapse_niimg(imgs, generate_sessions=False):
    data = []
    if generate_sessions:
        sessions = []
    first_img = iter(imgs).next()
    affine = first_img.get_affine()
    for index, iter_img in enumerate(imgs):
        img = check_niimg(iter_img)
        if not np.array_equal(img.get_affine(), affine):
            s_error = ""
            if generate_sessions:
                s_error = " of session #" + str(index)
            if (isinstance(iter_img, types.StringTypes)):
                i_error = "image " + iter_img
            else:
                i_error = "image #" + str(index)

            raise ValueError("Affine of %s%s is different"
                    " from reference affine"
                    "\nReference affine:\n%s\n"
                    "Wrong affine:\n%s"
                    % i_error, s_error,
                    repr(affine), repr(img.get_affine()))
        data.append(img)
        if generate_sessions:
            sessions += list(itertools.repeat(index, img.get_data().shape[-1]))
    if generate_sessions:
        return data, affine, sessions
    return data, affine


class MRITransformer(BaseEstimator, TransformerMixin):
    """MRI data loader with preprocessing

    Parameters
    ----------
    mask: 3D numpy matrix, optional
        Mask of the data. If not given, a mask is computed in the fit step.

    smooth: False or float, optional
        If smooth is not False, it gives the size, in voxel of the
        spatial smoothing to apply to the signal.

    confounds: CSV file path or 2D matrix

    detrend: boolean, optional

    verbose: interger, optional
        Indicate the level of verbosity. By default, nothing is printed
    """

    def __init__(self, sessions=None, mask=None, mask_connected=False,
            mask_opening=False, mask_lower_cutoff=0.2, mask_upper_cutoff=0.9,
            smooth=False, confounds=None, detrend=False,
            affine=None, low_pass=None, high_pass=None, t_r=None, copy=False,
            memory=Memory(cachedir=None, verbose=0), verbose=0):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        self.mask = mask
        self.mask_connected = mask_connected
        self.mask_opening = mask_opening
        self.mask_lower_cutoff = mask_lower_cutoff
        self.mask_upper_cutoff = mask_upper_cutoff
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend
        self.new_affine = affine
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.copy = copy
        self.memory = memory
        self.verbose = verbose
        self.sessions_ = sessions

    def load_imgs(self, imgs):
        # Initialization: we go through the data to get the depth and dimension
        depth = 0
        first_img = imgs
        while isinstance(first_img, collections.Iterable) \
                and not isinstance(first_img, types.StringTypes):
            first_img = iter(first_img).next()
            depth += 1

        # First Image is supposed to be a path or a Nifti like element
        first_img = check_niimg(first_img)

        # Check dimension and depth
        dim = len(first_img.get_data().shape)

        if not dim in [3, 4]:
            raise ValueError("[%s] Image must be a 3D or 4D array."
                    " Given image is a %iD array"
                    % self.__class__.__name__, dim)

        # Contains lengths of sessions to generate if needed
        if self.sessions_ is None and (dim + depth == 5):
            # With 4D images, each image is a session, we collapse them
            if dim == 4:
                imgs, affine, self.sessions_ = collapse_niimg(imgs,
                        generate_sessions=True)
            else:
                # We collapse the array and generate sessions
                lengths = [[i] * len(array) for i, array in enumerate(imgs)]
                self.sessions_ = list(itertools.chain.from_iterable(lengths))
                imgs = list(itertools.chain.from_iterable(imgs))
            # Remove the dimension corresponding to sessions
            dim -= 1

        if (dim + depth) != 4:
            # This error message is poor but givin details about each case
            # would be too complicated
            raise ValueError("[%s] Cannot load your data due to a dimension"
                    " problem." % self.__class__.__name__)

        # Now, we that data is in a known format, load it
        if dim == 4:
            data = check_niimg(imgs)
            affine = data.get_affine()
            data = data.get_data()
        else:
            data, affine = collapse_niimg(imgs)
        self.affine = affine
        return data

    def fit(self, X, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        X: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        memory = self.memory
        if isinstance(memory, basestring):
            memory = Memory(cachedir=memory)

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data" % self.__class__.__name__
        data = self.load_imgs(X)

        # Compute the mask if not given by the user
        if self.mask is None:
            if self.verbose > 0:
                print "[%s.fit] Computing the mask" % self.__class__.__name__
            self.mask_ = masking.compute_epi_mask(np.mean(data, axis=-1),
                    connected=self.mask_connected, opening=self.mask_opening,
                    lower_cutoff=self.mask_lower_cutoff,
                    upper_cutoff=self.mask_upper_cutoff)
        else:
            if isinstance(self.mask, types.StringTypes):
                self.mask_ = nibabel.load(self.mask).get_data() \
                    .astype(np.bool)
            else:
                self.mask_ = self.mask

        return self

    def transform(self, X):
        memory = self.memory

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[%s.fit] Loading data" % self.__class__.__name__
        data = self.load_imgs(X)
        affine = self.affine

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[%s.transform] Resampling" % self.__class__.__name__
        data, affine = memory.cache(resampling.as_volume_img)(data, affine,
                new_affine=self.new_affine, copy=self.copy)

        # Function that does that exposes interpolation order, but not
        # this object
        # XXX -> ?

        # Get series from data with optional smoothing
        if self.verbose > 0:
            print "[%s.transform] Masking and smoothing" \
                % self.__class__.__name__
        data = masking.extract_time_series(data, affine,
                self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 0:
            print "[%s.transform] Cleaning signal" % self.__class__.__name__
        if self.sessions_ is None:
            data = memory.cache(preprocessing.clean_signals)(data,
                    confounds=self.confounds, low_pass=self.low_pass,
                    high_pass=self.high_pass, t_r=self.t_r,
                    detrend=self.detrend, normalize=False)
        else:
            for s in np.unique(self.sessions_):
                if self.confounds is not None:
                    session_confounds = self.confounds[self.sessions_ == s]
                    data[self.sessions_ == s] = \
                        memory.cache(preprocessing.clean_signals)(
                                data=data[self.sessions_ == s],
                                confounds=session_confounds,
                                low_pass=self.low_pass,
                                high_pass=self.high_pass, t_r=self.t_r,
                                detrend=self.detrend, normalize=False)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        # data is in format voxel x time_series. We inverse it
        data = np.rollaxis(data, -1)

        self.affine_ = affine
        return data

    def inverse_transform(self, X, null=-1):
        if len(X.shape) > 1:
            shape = self.mask_.shape + (X.shape[-1],)
        else:
            shape = self.mask_.shape
        data = np.empty(shape)
        data.fill(null)
        data[self.mask_] = np.rollaxis(X, -1)
        return utils.Niimg(data, self.affine)
