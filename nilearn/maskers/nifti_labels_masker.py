"""
Transformer for computing ROI signals.
"""

import numpy as np
import warnings

from joblib import Memory

from nilearn import _utils, image, masking
from nilearn.maskers.base_masker import _filter_and_extract, BaseMasker


class _ExtractionFunctor(object):

    func_name = 'nifti_labels_masker_extractor'

    def __init__(self, _resampled_labels_img_, background_label, strategy,
                 mask_img):
        self._resampled_labels_img_ = _resampled_labels_img_
        self.background_label = background_label
        self.strategy = strategy
        self.mask_img = mask_img

    def __call__(self, imgs):
        from ..regions import signal_extraction

        return signal_extraction.img_to_signals_labels(
            imgs, self._resampled_labels_img_,
            background_label=self.background_label, strategy=self.strategy,
            mask_img=self.mask_img)


@_utils.fill_doc
class NiftiLabelsMasker(BaseMasker, _utils.CacheMixin):
    """Class for masking of Niimg-like objects.

    NiftiLabelsMasker is useful when data from non-overlapping volumes should
    be extracted (contrarily to :class:`nilearn.maskers.NiftiMapsMasker`).
    Use case: Summarize brain signals from clusters that were obtained by prior
    K-means or Ward clustering.

    Parameters
    ----------
    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image. This is used
        to improve reporting quality if provided.
        Warning: The labels must be consistent with the label
        values provided through `labels_img`.

    background_label : :obj:`int` or :obj:`float`, optional
        Label used in labels_img to represent background.
        Warning: This value must be consistent with label values and
        image provided.
        Default=0.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.
    %(smoothing_fwhm)s
    standardize : {False, True, 'zscore', 'psc'}, optional
        Strategy to standardize the signal.
        'zscore': the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        'psc':  Timeseries are shifted to zero mean value and scaled
        to percent signal change (as compared to original mean signal).
        True : the signal is z-scored. Timeseries are shifted
        to zero mean and scaled to unit variance.
        False : Do not standardize the data.
        Default=False.

    standardize_confounds : :obj:`bool`, optional
        If standardize_confounds is True, the confounds are z-scored:
        their mean is put to 0 and their variance to 1 in the time dimension.
        Default=True.

    high_variance_confounds : :obj:`bool`, optional
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out. Default=False.

    detrend : :obj:`bool`, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details. Default=False.

    low_pass : None or :obj:`float`, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    high_pass : None or :obj:`float`, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    t_r : :obj:`float`, optional
        This parameter is passed to signal.clean. Please see the related
        documentation for details

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    resampling_target : {"data", "labels", None}, optional
        Gives which image gives the final shape/size. For example, if
        `resampling_target` is "data", the atlas is resampled to the
        shape of the data if needed. If it is "labels" then mask_img
        and images provided to fit() are resampled to the shape and
        affine of maps_img. "None" means no resampling: if shapes and
        affines do not match, a ValueError is raised. Default="data".

    memory : :obj:`joblib.Memory` or :obj:`str`, optional
        Used to cache the region extraction process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : :obj:`int`, optional
        Aggressiveness of memory caching. The higher the number, the higher
        the number of functions that will be cached. Zero means no caching.
        Default=1.

    verbose : :obj:`int`, optional
        Indicate the level of verbosity. By default, nothing is printed
        Default=0.

    strategy : :obj:`str`, optional
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation. Default='mean'.

    reports : :obj:`bool`, optional
        If set to True, data is saved in order to produce a report.
        Default=True.

    Attributes
    ----------
    mask_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The mask of the data, or the computed one.

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image.

    n_elements_ : :obj:`int`
        The number of discrete values in the mask.
        This is equivalent to the number of unique values in the mask image,
        ignoring the background value.

        .. versionadded:: 0.9.2

    See also
    --------
    nilearn.maskers.NiftiMasker

    """
    # memory and memory_level are used by _utils.CacheMixin.

    def __init__(
        self, labels_img,
        labels=None,
        background_label=0,
        mask_img=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        high_variance_confounds=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        dtype=None,
        resampling_target='data',
        memory=Memory(location=None, verbose=0),
        memory_level=1,
        verbose=0,
        strategy='mean',
        reports=True,
    ):
        self.labels_img = labels_img
        self.labels = labels
        self.background_label = background_label
        self.mask_img = mask_img

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

        # Parameters for resampling
        self.resampling_target = resampling_target

        # Parameters for joblib
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.reports = reports
        self._report_content = dict()
        self._report_content['description'] = (
            'This reports shows the regions defined by the labels of the mask.'
        )
        self._report_content['warning_message'] = None

        available_reduction_strategies = {
            'mean',
            'median',
            'sum',
            'minimum',
            'maximum',
            'standard_deviation',
            'variance',
        }

        if strategy not in available_reduction_strategies:
            raise ValueError(
                f"Invalid strategy '{strategy}'. "
                f"Valid strategies are {available_reduction_strategies}."
            )

        self.strategy = strategy

        if resampling_target not in ('labels', 'data', None):
            raise ValueError(
                "invalid value for 'resampling_target' "
                f"parameter: {resampling_target}"
            )

    def generate_report(self):
        from nilearn.reporting.html_report import generate_report
        return generate_report(self)

    def _reporting(self):
        """
        Returns
        -------
        displays : list
            A list of all displays to be rendered.

        """
        try:
            import matplotlib.pyplot as plt
            from nilearn import plotting
        except ImportError:
            with warnings.catch_warnings():
                mpl_unavail_msg = (
                    'Matplotlib is not imported! No reports will be generated.'
                )
                warnings.filterwarnings('always', message=mpl_unavail_msg)
                warnings.warn(category=ImportWarning, message=mpl_unavail_msg)
                return [None]

        if self._reporting_data is not None:
            labels_image = self._reporting_data['labels_image']
        else:
            labels_image = None

        if labels_image is not None:
            # Remove warning message in case where the masker was
            # previously fitted with no func image and is re-fitted
            if 'warning_message' in self._report_content:
                self._report_content['warning_message'] = None

            labels_image = image.load_img(labels_image, dtype='int32')
            labels_image_data = image.get_data(labels_image)
            labels_image_affine = labels_image.affine
            # Number of regions excluding the background
            number_of_regions = np.sum(np.unique(labels_image_data)
                                       != self.background_label)
            # Basic safety check to ensure we have as many labels as we
            # have regions (plus background).
            if(self.labels is not None
               and len(self.labels) != number_of_regions + 1):
                raise ValueError(
                    'Mismatch between the number of provided labels '
                    f'({len(self.labels)}) and the number of regions in '
                    f'provided label image ({number_of_regions + 1}).'
                )

            self._report_content['number_of_regions'] = number_of_regions

            label_values = np.unique(labels_image_data)
            label_values = label_values[label_values != self.background_label]
            columns = [
                'label value',
                'region name',
                'size (in mm^3)',
                'relative size (in %)',
            ]

            if self.labels is None:
                columns.remove('region name')

            regions_summary = {c: [] for c in columns}
            for label in label_values:
                regions_summary['label value'].append(label)
                if self.labels is not None:
                    regions_summary['region name'].append(self.labels[label])

                size = len(labels_image_data[labels_image_data == label])
                voxel_volume = np.abs(np.linalg.det(
                    labels_image_affine[:3, :3]))
                regions_summary['size (in mm^3)'].append(round(
                    size * voxel_volume))
                regions_summary['relative size (in %)'].append(round(
                    size / len(
                        labels_image_data[labels_image_data != 0]
                    ) * 100, 2))

            self._report_content['summary'] = regions_summary

            img = self._reporting_data['img']
            # If we have a func image to show in the report, use it
            if img is not None:
                dim = image.load_img(img).shape
                if len(dim) == 4:
                    # compute middle image from 4D series for plotting
                    img = image.index_img(img, dim[-1] // 2)
                display = plotting.plot_img(
                    img,
                    black_bg=False,
                    cmap='CMRmap_r',
                )
                plt.close()
                display.add_contours(labels_image, filled=False, linewidths=3)

            # Otherwise, simply plot the ROI of the label image
            # and give a warning to the user
            else:
                msg = (
                    'No image provided to fit in NiftiLabelsMasker. '
                    'Plotting ROIs of label image on the '
                    'MNI152Template for reporting.'
                )
                warnings.warn(msg)
                self._report_content['warning_message'] = msg
                display = plotting.plot_roi(labels_image)
                plt.close()

            # If we have a mask, show its contours
            if self._reporting_data['mask'] is not None:
                display.add_contours(
                    self._reporting_data['mask'],
                    filled=False,
                    colors="g",
                    linewidths=3,
                )
        else:
            self._report_content['summary'] = None
            display = None

        return [display]

    def fit(self, imgs=None, y=None):
        """Prepare signal extraction from regions.

        All parameters are unused, they are for scikit-learn compatibility.

        """
        _utils.logger.log(
            'loading data from %s' % _utils._repr_niimgs(
                self.labels_img,
                shorten=(not self.verbose),
            ),
            verbose=self.verbose,
        )
        self.labels_img_ = _utils.check_niimg_3d(self.labels_img)
        if self.mask_img is not None:
            _utils.logger.log(
                'loading data from %s' % _utils._repr_niimgs(
                    self.mask_img,
                    shorten=(not self.verbose),
                ),
                verbose=self.verbose,
            )
            self.mask_img_ = _utils.check_niimg_3d(self.mask_img)

        else:
            self.mask_img_ = None

        # Check shapes and affines or resample.
        if self.mask_img_ is not None:
            if self.resampling_target == 'data':
                # resampling will be done at transform time
                pass

            elif self.resampling_target is None:
                if self.mask_img_.shape != self.labels_img_.shape[:3]:
                    raise ValueError(
                        _utils._compose_err_msg(
                            'Regions and mask do not have the same shape',
                            mask_img=self.mask_img,
                            labels_img=self.labels_img,
                        )
                    )

                if not np.allclose(
                    self.mask_img_.affine,
                    self.labels_img_.affine,
                ):
                    raise ValueError(
                        _utils._compose_err_msg(
                            'Regions and mask do not have the same affine.',
                            mask_img=self.mask_img,
                            labels_img=self.labels_img,
                        ),
                    )

            elif self.resampling_target == 'labels':
                _utils.logger.log('resampling the mask', verbose=self.verbose)
                self.mask_img_ = image.resample_img(
                    self.mask_img_,
                    target_affine=self.labels_img_.affine,
                    target_shape=self.labels_img_.shape[:3],
                    interpolation='nearest',
                    copy=True)

            else:
                raise ValueError(
                    'Invalid value for '
                    f'resampling_target: {self.resampling_target}'
                )

            # Just check that the mask is valid
            masking._load_mask_img(self.mask_img_)

        if not hasattr(self, '_resampled_labels_img_'):
            # obviates need to run .transform() before .inverse_transform()
            self._resampled_labels_img_ = self.labels_img_

        if self.reports:
            self._reporting_data = {
                'labels_image': self._resampled_labels_img_,
                'mask': self.mask_img_,
                'img': imgs,
            }
        else:
            self._reporting_data = None

        # Infer the number of elements in the mask
        # This is equal to the number of unique values in the label image,
        # minus the background value.
        self.n_elements_ = np.unique(
            image.get_data(self._resampled_labels_img_)
        ).size - 1

        return self

    def fit_transform(self, imgs, confounds=None, sample_mask=None):
        """Prepare and perform signal extraction from regions.

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
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D :obj:`numpy.ndarray`
            Signal for each label.
            shape: (number of scans, number of labels)

        """
        return self.fit().transform(imgs, confounds=confounds,
                                    sample_mask=sample_mask)

    def _check_fitted(self):
        if not hasattr(self, 'labels_img_'):
            raise ValueError(
                f'It seems that {self.__class__.__name__} has not been '
                'fitted. '
                'You must call fit() before calling transform().'
            )

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
            shape: (number of scans - number of volumes removed, )
            Masks the niimgs along time/fourth dimension to perform scrubbing
            (remove volumes with high motion) and/or non-steady-state volumes.
            This parameter is passed to signal.clean.

                .. versionadded:: 0.8.0

        Returns
        -------
        region_signals : 2D numpy.ndarray
            Signal for each label.
            shape: (number of scans, number of labels)

        Warns
        -----
        DeprecationWarning
            If a 3D niimg input is provided, the current behavior
            (adding a singleton dimension to produce a 2D array) is deprecated.
            Starting in version 0.12, a 1D array will be returned for 3D
            inputs.

        """
        # We handle the resampling of labels separately because the affine of
        # the labels image should not impact the extraction of the signal.

        if not hasattr(self, '_resampled_labels_img_'):
            self._resampled_labels_img_ = self.labels_img_

        if not hasattr(self, '_resampled_mask_img'):
            self._resampled_mask_img = self.mask_img_

        if self.resampling_target == "data":
            imgs_ = _utils.check_niimg(imgs, atleast_4d=True)
            if not _utils.niimg_conversions._check_same_fov(
                imgs_,
                self._resampled_labels_img_,
            ):
                if self.verbose > 0:
                    print("Resampling labels")
                labels_before_resampling = set(
                    np.unique(
                        _utils.niimg._safe_get_data(
                            self._resampled_labels_img_,
                        )
                    )
                )
                self._resampled_labels_img_ = self._cache(
                    image.resample_img, func_memory_level=2)(
                        self.labels_img_, interpolation="nearest",
                        target_shape=imgs_.shape[:3],
                        target_affine=imgs_.affine)
                labels_after_resampling = set(
                    np.unique(
                        _utils.niimg._safe_get_data(
                            self._resampled_labels_img_,
                        )
                    )
                )
                labels_diff = labels_before_resampling.difference(
                    labels_after_resampling
                )
                if len(labels_diff) > 0:
                    warnings.warn("After resampling the label image to the "
                                  "data image, the following labels were "
                                  f"removed: {labels_diff}. "
                                  "Label image only contains "
                                  f"{len(labels_after_resampling)} labels "
                                  "(including background).")

            if (self.mask_img is not None) and (
                not _utils.niimg_conversions._check_same_fov(
                    imgs_,
                    self._resampled_mask_img,
                )
            ):
                if self.verbose > 0:
                    print("Resampling mask")
                self._resampled_mask_img = self._cache(
                    image.resample_img, func_memory_level=2)(
                        self.mask_img_, interpolation="nearest",
                        target_shape=imgs_.shape[:3],
                        target_affine=imgs_.affine)

            # Remove imgs_ from memory before loading the same image
            # in filter_and_extract.
            del imgs_

        target_shape = None
        target_affine = None
        if self.resampling_target == 'labels':
            target_shape = self._resampled_labels_img_.shape[:3]
            target_affine = self._resampled_labels_img_.affine

        params = _utils.class_inspect.get_params(
            NiftiLabelsMasker,
            self,
            ignore=['resampling_target'],
        )
        params['target_shape'] = target_shape
        params['target_affine'] = target_affine

        region_signals, labels_ = self._cache(
            _filter_and_extract,
            ignore=['verbose', 'memory', 'memory_level'],
        )(
            # Images
            imgs, _ExtractionFunctor(
                self._resampled_labels_img_,
                self.background_label,
                self.strategy,
                self._resampled_mask_img,
            ),
            # Pre-processing
            params,
            confounds=confounds,
            sample_mask=sample_mask,
            dtype=self.dtype,
            # Caching
            memory=self.memory,
            memory_level=self.memory_level,
            verbose=self.verbose,
        )

        self.labels_ = labels_

        return region_signals

    def inverse_transform(self, signals):
        """Compute voxel signals from region signals

        Any mask given at initialization is taken into account.

        .. versionchanged:: 0.9.2dev

            This method now supports 1D arrays, which will produce 3D images.

        Parameters
        ----------
        signals : 1D/2D :obj:`numpy.ndarray`
            Signal for each region.
            If a 1D array is provided, then the shape should be
            (number of elements,), and a 3D img will be returned.
            If a 2D array is provided, then the shape should be
            (number of scans, number of elements), and a 4D img will be
            returned.

        Returns
        -------
        img : :obj:`nibabel.nifti1.Nifti1Image`
            Signal for each voxel
            shape: (X, Y, Z, number of scans)

        """
        from ..regions import signal_extraction

        self._check_fitted()

        _utils.logger.log("computing image from signals", verbose=self.verbose)
        return signal_extraction.signals_to_img_labels(
            signals,
            self._resampled_labels_img_,
            self.mask_img_,
            background_label=self.background_label,
        )
