"""
Transformer used to apply basic tranformations on MRI data.
"""

from sklearn.base import BaseEstimator, TransformerMixin

# Note: every input here is either a filename or a NiImage

# check_niimg: utility function that loads if given filename, else
# check 'get_data', 'get_affine'

import types
import nibabel
import masking
import resampling
import preprocessing
import numpy as np


def check_niimg(data):
    if isinstance(data, types.StringTypes):
        # data is a filename, we load it
        result = nibabel.load(data)
    else:
        # it is an object, it should have get_data and get_affine methods
        # list the methods of the object
        methods = [method for method in dir(data)
                if callable(getattr(data, method))]
        if not ('get_data' in methods and 'get_affine' in methods):
            raise AttributeError('missing get_data or get_affine method')
        result = data
    return result


# Rmk: needs a similar object for ROIs or not

class NiftiLoader(BaseEstimator, TransformerMixin):

    def __init__(self, mask=None, smooth=False, confounds=None, detrend=False):
        # Mask is compulsory or computed
        # Must integrate memory, as in WardAgglomeration
        self.mask_ = mask
        self.smooth = smooth
        self.confounds = confounds
        self.detrend = detrend

    def fit(self, X, y=None):
        """
        X: list of filenames or NiImages
        If this is a list, the affine is the same for all
        """

        # Loading
        # =======
        # If filenames are given, load them. Otherwise just take original
        # data.

        print "Loading"
        if (isinstance(X, np.ndarray)):
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

        # Spatial
        # =======
        #
        # optional compute_mask (nipy.labs.mask.compute_mask_files)

        print "Computing mask"
        if self.mask_ is None:
            self.mask_ = masking.compute_mask(np.mean(data, axis=-1))

        # resampling:
        # nipy.labs.datasets.volumes.VolumeImg.as_volume_img
        # affine is either 4x4, 3x3

        print "Resampling"
        resampled, affine = resampling.as_volume_img(data,
                affine=affine, copy=False)

        # Function that does that exposes interpolation order, but not
        # this object
        # optional extract series (apply mask: nipy.labs.mask.series_from_mask)
        #           -> smoothing

        print "Masking and smoothing"
        series = masking.series_from_mask(resampled, affine,
                mask, smooth=self.smooth)

        # Temporal
        # ========
        # (parietal-python) parietal.time_series.preprocessing.clean_signals
        # Detrending (optional)
        # Filtering (grab TR from header)
        # Confounds (from csv file or numpy array)
        # Normalizing

        print "Cleaning signal"
        signals = preprocessing.clean_signals(series, confounds=self.confounds,
                detrend=self.detrend)

        # For _later_: missing value removal or imputing of missing data
        # (i.e. we want to get rid of NaNs, if smoothing must be done
        # earlier)
        # Optionally: 'doctor_nan', remove voxels with NaNs, other option
        # for later: some form of imputation

        return signals
