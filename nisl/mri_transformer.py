"""
Transformer used to apply basic tranisformations on MRI data.
"""

import types

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import nibabel

from . import masking
from . import resampling
from . import preprocessing


def _check_callable_method(object, method_name):
    # get the attribute, raise an error if missing
    attr = getattr(object, method_name)
    if not callable(attr):
        raise AttributeError(method_name + ' is not a valid method')


def check_niimg(data):
    if isinstance(data, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(data)
    else:
        # it is an object, it should have get_data and get_affine methods
        _check_callable_method(data, 'get_data')
        _check_callable_method(data, 'get_affine')
        result = data
    return result


# Rmk: needs a similar object for ROIs or not

class MRItransformer(BaseEstimator, TransformerMixin):
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

    def __init__(self, mask=None, smooth=False, confounds=None, detrend=False,
            affine=None, low_pass=0.2, high_pass=None, t_r=2.5, verbose=0):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        self.mask_ = mask
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend
        self.affine = affine
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.verbose = verbose

    def fit(self, X, y=None):
        """Compute the mask corresponding to the data

        Parameters
        ----------
        X: list of filenames or NiImages
            Data on which the mask must be calculated. If this is a list,
            the affine is considered the same for all.
        """

        # Load data (if filenames are given, load them)
        if self.verbose > 0:
            print "[MRITransformer.fit] Loading data"
        if (isinstance(X, types.StringTypes)):
            data = []
            affine = None
            # Roll axis to index by scan
            scans = np.roll_axis(X, -1)
            for scan in scans:
                img = check_niimg(scan)
                if affine is None:
                    affine = img.get_affine()
                data.append(img.get_data())
                del img
            data = np.asarray(data)
            np.rollaxis(data, 0, start=4)
        else:
            img = check_niimg(X)
            affine = img.get_affine()
            data = img.get_data()

        # Compute the mask if not given by the user

        if self.mask_ is None:
            if self.verbose > 0:
                print "[MRITransformer.fit] Computing the mask"
            self.mask_ = masking.compute_epi_mask(np.mean(data, axis=-1))

        return self

    def transform(self, X):

        # Load data (if filenames are given, load them)
        # XXX This should be done once and for all, to in fit and transform...
        if self.verbose > 0:
            print "[MRITransformer.fit] Loading data"
        if (isinstance(X, types.StringTypes)):
            data = []
            affine = None
            # Roll axis to index by scan
            scans = np.roll_axis(X, -1)
            for scan in scans:
                img = check_niimg(scan)
                if affine is None:
                    affine = img.get_affine()
                data.append(img.get_data())
                del img
            data = np.asarray(data)
            np.rollaxis(data, 0, start=4)
        else:
            img = check_niimg(X)
            affine = img.get_affine()
            data = img.get_data()

        # Resampling: allows the user to change the affine, the shape or both
        if self.verbose > 0:
            print "[MRITransformer.transform] Resampling"
        data, affine = resampling.as_volume_img(data, affine,
                new_affine=self.affine, copy=False)

        # Function that does that exposes interpolation order, but not
        # this object
        # XXX -> ?

        # Get series from data with optional smoothing
        if self.verbose > 0:
            print "[MRITransformer.transform] Masking and smoothing"
        data = masking.series_from_mask(data, affine,
                self.mask_, smooth=self.smooth)

        # Temporal
        # ========
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        if self.verbose > 0:
            print "[MRITransformer.transform] Cleaning signal"
        data = preprocessing.clean_signals(data, confounds=self.confounds,
                low_pass=self.low_pass, high_pass=self.high_pass, t_r=self.t_r,
                detrend=self.detrend)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        return (data, affine)

    def inverse_transform(self, X):
        # From masked data to np.masked_array
        # shape = self.mask_.shape + (X.shape[3],)
        pass
