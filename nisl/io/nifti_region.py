"""
Transformer for computing ROI signals.
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals.joblib import Memory

import nibabel

from .. import utils
from .. import signals
from .. import region
from .. import masking


class NiftiLabelsMasker(BaseEstimator, TransformerMixin):
    """Extract labeled-defined region signals from images.

    Parameters
    ==========
    labels_img (niimg)
        Region definitions, as one image of labels.

    background_label (number)
        Label used in labels_img to represent background.

    mask_img (niimg)
        Mask to apply to regions before extracting signals.

    memory (joblib.Memory or str)
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level (int, optional)
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.

    See also
    ========
    nisl.io.NiftiMasker
    """

    def __init__(self, labels_img, background_label=0, mask_img=None,
                 smooth=None, standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 memory=Memory(cachedir=None, verbose=0), memory_level=0,
                 verbose=0):
        self.labels_img = labels_img
        self.background_label = background_label
        self.mask_img = mask_img

        # Parameters for signals_from_regions()
        self.smooth = smooth

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

    def fit(self, y=None):
        """Prepare signal extraction from regions.
        """
        self.labels_img_ = utils.check_niimg(self.labels_img)
        if self.mask_img is not None:
            mask_data, mask_affine = masking._load_mask_img(self.mask_img)
            if mask_data.shape != self.labels_img_.shape[:3]:
                raise ValueError("Regions and mask do not have the same shape")
            if abs(mask_affine - self.labels_img_.get_affine()).max() > 1e-9:
                raise ValueError("Regions and mask do not have the same "
                                 "affine.")
            # This may not work with all niimgs. In some, get_data() may return
            # a copy of the underlying object.
            self.labels_img_.get_data(
                )[np.logical_not(mask_data)] = self.background_label

        return self

    def transform(self, niimgs, confounds=None):
        """Extract signals from images.

        Parameters
        ==========
        niimgs (niimg)
            Images to process. It must boils down to a 4D image with scans
            number as last dimension.
        confounds (array-like)
            This parameter is passed to signal.clean. Please see the related
            documentation for details.
            shape: (number of scans, number of confounds)

        Returns
        =======
        signals (2D numpy.ndarray)
            Signal for each region.
            shape: (number of scans, number of regions)

        """
        niimgs = utils.check_niimgs(niimgs)
        data = utils.as_ndarray(niimgs.get_data())
        affine = niimgs.get_affine()
        if self.smooth is not None:
            # FIXME: useless copy if input parameter niimg is a string.
            data = masking._smooth_array(data, affine, smooth=self.smooth,
                                         copy=True)

        region_signals, self.labels_ = region.img_to_signals_labels(
            nibabel.Nifti1Image(data, affine), self.labels_img_,
            background_label=self.background_label)

        region_signals = signals.clean(region_signals,
                                       detrend=self.detrend,
                                       standardize=self.standardize,
                                       t_r=self.t_r,
                                       low_pass=self.low_pass,
                                       high_pass=self.high_pass,
                                       confounds=confounds)
        # FIXME: put into signals.clean()
        region_signals /= region_signals.std(axis=0)
        return region_signals

    def inverse_transform(self, signals):
        """Compute voxel signals from region signals

        Parameters
        ==========
        signals (2D numpy.ndarray)
            Signal for each region.
            shape: (number of scans, number of regions)

        Returns
        =======
        voxel_signals (Nifti1Image)
            Signal for each voxel
            shape: (number of scans, number of voxels)
        """
        return region.signals_to_img_labels(signals, self.labels_img_,
                                        background_label=self.background_label)
